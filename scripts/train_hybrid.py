"""

Train SLM as well as logistic regression.

Test locally with CPU
```
python scripts/train_slm_logistic_reg.py --cpu \
--dataset data/MNIST_no_psf_down6_1000files --batch_size 20
```

On server
```
python scripts/train_slm_logistic_reg.py  \
--dataset data/MNIST_no_psf_down2 --batch_size 50 \
--mean 0.0011 --std 0.0290
```

"""

from curses import meta
from lenslessclass.models import SLMMultiClassLogistic
import torch
import random
from pprint import pprint
import pathlib as plib
import os
import torch.nn as nn
from lenslessclass.datasets import MNISTAugmented
import torch.optim as optim
import torchvision.transforms as transforms
import time
import click
from waveprop.devices import SLMOptions, SensorOptions, slm_dict, sensor_dict, SensorParam
import numpy as np
import json
from os.path import dirname, abspath, join
from datetime import datetime
from lenslessclass.datasets import simulate_propagated_dataset


@click.command()
@click.option(
    "--dataset",
    type=str,
    help="Path to dataset.",
)
@click.option(
    "--cont",
    type=str,
    help="Path to training to continue.",
)
@click.option("--seed", type=int, help="Random seed.", default=0)
@click.option("--slm", type=str, help="Which SLM to use.", default=SLMOptions.ADAFRUIT.value)
@click.option("--sensor", type=str, help="Which sensor to use.", default=SensorOptions.RPI_HQ.value)
@click.option(
    "--crop_fact",
    type=float,
    default=0.7,
    help="Fraction of sensor that is left uncropped, centered.",
)
@click.option(
    "--output_dir", type=str, default="data", help="Path to save augmented dataset (if created)."
)
@click.option("--simple", is_flag=True, help="Don't take into account deadspace.")
@click.option("--scene2mask", type=float, default=0.4, help="Scene to SLM/mask distance in meters.")
@click.option(
    "--mask2sensor", type=float, default=0.004, help="SLM/mask to sensor distance in meters."
)
@click.option("--object_height", type=float, default=0.12, help="Object height in meters.")
@click.option("--lr", type=float, help="Learning rate for SGD.", default=0.01)
@click.option("--momentum", type=float, help="Momentum for SGD.", default=0.01)
@click.option("--n_epoch", type=int, help="Number of epochs to train.", default=10)
@click.option("--batch_size", type=int, help="Batch size.", default=30)
@click.option(
    "--print_epoch",
    type=int,
    help="How many batches to wait before printing epoch progress.",
    default=None,
)
@click.option(
    "--mean",
    type=float,
    help="Mean of original dataset to normalize, if not provided it will be computed.",
)
@click.option(
    "--sensor_act",
    type=str,
    help="Activation at sensor. If not provided, none will applied.",
    default=None,
)
@click.option(
    "--dropout",
    default=None,
    type=float,
    help="Percentage of dropout after diffractive optical layer.",
)
@click.option(
    "--opti",
    type=click.Choice(["sgd", "adam"], case_sensitive=False),
    help="Optimizer.",
    default="adam",
)
@click.option(
    "--std",
    type=float,
    help="Standard deviation of original dataset to normalize, if not provided it will be computed.",
)
@click.option(
    "--noise_type",
    default=None,
    type=click.Choice(["speckle", "gaussian", "s&p", "poisson"]),
    help="Noise type to add.",
)
@click.option("--snr", default=40, type=float, help="SNR to determine noise to add.")
@click.option("--down_out", type=float, help="Factor by which to downsample output.", default=128)
@click.option(
    "--down_psf", type=float, help="Factor by which to downsample convolution.", default=8
)
@click.option(
    "--output_dim",
    default=None,
    nargs=2,
    type=int,
    help="Output dimension (height, width). Use this instead of `down_out` if provided",
)
@click.option("--n_files", type=int, default=None)
@click.option("--device", type=str, help="Main device for training.")
@click.option(
    "--single_gpu",
    is_flag=True,
    help="Whether to use single GPU is multiple available. Default will try using all.",
)
def train_slm_logistic_reg(
    dataset,
    slm,
    sensor,
    crop_fact,
    simple,
    scene2mask,
    mask2sensor,
    lr,
    momentum,
    n_epoch,
    batch_size,
    mean,
    std,
    down_out,
    down_psf,
    print_epoch,
    sensor_act,
    opti,
    dropout,
    seed,
    noise_type,
    snr,
    output_dim,
    cont,
    object_height,
    output_dir,
    n_files,
    device,
    single_gpu,
):

    if cont:
        cont = plib.Path(cont)
        print(f"\nCONTINUTING TRAINING FOR {n_epoch} EPOCHS")
        f = open(str(cont / "metadata.json"))
        metadata = json.load(f)
        pprint(metadata)

        dataset = metadata["dataset"]
        mean = metadata["mean"]
        std = metadata["std"]
        slm = metadata["slm"]
        seed = metadata["seed"]
        sensor = metadata["sensor"]
        sensor_act = metadata["sensor_activation"]
        batch_size = metadata["batch_size"]
        noise_type = metadata["noise_type"]
        snr = metadata["snr"]
        input_shape = np.array(metadata["model_param"]["input_shape"])
        crop_fact = metadata["model_param"]["crop_fact"]
        device = metadata["model_param"]["device"]
        simple = not metadata["model_param"]["deadspace"]
        scene2mask = metadata["model_param"]["scene2mask"]
        mask2sensor = metadata["model_param"]["mask2sensor"]
        device_mask_creation = metadata["model_param"]["device_mask_creation"]
        output_dim = metadata["model_param"]["output_dim"]
        multi_gpu = metadata["model_param"]["multi_gpu"]
        dropout = metadata["model_param"]["dropout"]

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    device_mask_creation = "cpu"  # TODO: bc doesn't fit on GPU

    if print_epoch is None:
        print_epoch = batch_size

    sensor_act_fn = None
    if sensor_act is not None:
        # https://pytorch.org/docs/stable/nn.functional.html#non-linear-activation-functions
        if sensor_act == "relu":
            sensor_act_fn = nn.ReLU()
        elif sensor_act == "leaky":
            sensor_act_fn = nn.LeakyReLU(float=0.1)
        elif sensor_act == "tanh":
            sensor_act_fn = nn.Tanh()
        else:
            raise ValueError("Not supported activation.")

    sensor_param = sensor_dict[sensor]

    if output_dim is None:
        if down_out:
            output_dim = tuple((sensor_param[SensorParam.SHAPE] * 1 / down_out).astype(int))
        else:
            output_dim = sensor_param[SensorParam.SHAPE]

    ## load mnist dataset
    use_cuda = torch.cuda.is_available()
    multi_gpu = False
    if device is None:
        if use_cuda:
            print("CUDA available, using GPU.")
            device = "cuda"
            n_gpus = torch.cuda.device_count()
            if n_gpus > 1 and not single_gpu:
                multi_gpu = True
                print(f"-- using {n_gpus} GPUs")
            else:
                multi_gpu = False

        else:
            device = "cpu"
            print("CUDA not available, using CPU.")
    else:
        if device == "cpu":
            use_cuda = False
        if use_cuda:
            n_gpus = torch.cuda.device_count()
            if n_gpus > 1 and not single_gpu:
                multi_gpu = True
                print(f"-- using {n_gpus} GPUs")

    # check if need to create dataset
    if dataset is None:
        dataset = simulate_propagated_dataset(
            dataset="MNIST",
            down_psf=down_psf,
            sensor=sensor,
            down_out=None,  # done during training
            scene2mask=scene2mask,
            mask2sensor=mask2sensor,
            object_height=object_height,
            device=device,
            crop_output=False,
            grayscale=True,
            single_psf=False,
            output_dir=output_dir,
            n_files=n_files,
        )

    # -- first determine mean and standard deviation (of training set)
    if mean is None and std is None:
        trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0, 1)])
        print("\nComputing stats...")
        train_set = MNISTAugmented(path=dataset, train=True, transform=trans)
        mean, std = train_set.get_stats(batch_size=batch_size)
        print("Dataset mean : ", mean)
        print("Dataset standard deviation : ", std)

        del train_set

    # -- normalize according to training set stats
    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    train_set = MNISTAugmented(path=dataset, train=True, transform=trans)
    test_set = MNISTAugmented(path=dataset, train=False, transform=trans)
    input_shape = train_set.output_dim

    train_loader = torch.utils.data.DataLoader(
        dataset=train_set, batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set, batch_size=batch_size, shuffle=False
    )

    print("==>>> total training batch number: {}".format(len(train_loader)))
    print("==>>> total testing batch number: {}".format(len(test_loader)))

    ## hybrid neural network
    model = SLMMultiClassLogistic(
        input_shape=input_shape,
        slm_config=slm_dict[slm],
        sensor_config=sensor_param,
        crop_fact=crop_fact,
        device=device,
        deadspace=not simple,
        scene2mask=scene2mask,
        mask2sensor=mask2sensor,
        device_mask_creation=device_mask_creation,
        output_dim=output_dim,
        sensor_activation=sensor_act_fn,
        multi_gpu=multi_gpu,
        dropout=dropout,
        noise_type=noise_type,
        snr=snr,
    )

    if use_cuda:
        model = model.to(device)

    if cont:
        state_dict_fp = str(cont / "state_dict.pth")
        model.load_state_dict(torch.load(state_dict_fp))
        model.compute_intensity_psf()

    # set optimizer
    # TODO : set different learning rates: https://pytorch.org/docs/stable/optim.html
    if opti == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    elif opti == "adam":
        # same default params
        optimizer = optim.Adam(
            model.parameters(),
            lr=0.001,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=0,
            amsgrad=False,
        )
    else:
        raise ValueError("Invalid optimization approach.")

    criterion = nn.CrossEntropyLoss()

    # Print model and optimizer state_dict
    print("\nModel's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    print()
    print("Optimizer's state_dict:")
    for var_name in optimizer.state_dict():
        print(var_name, "\t", optimizer.state_dict()[var_name])

    ## save best model param
    n_hidden = np.prod(output_dim)
    timestamp = datetime.now().strftime("%d%m%Y_%Hh%M")
    model_output_dir = f"./{os.path.basename(dataset)}_outdim{int(n_hidden)}_{n_epoch}epoch_seed{seed}_logistic_reg"
    if noise_type:
        model_output_dir += f"_{noise_type}{snr}"
    model_output_dir += f"_{timestamp}"

    model_output_dir = plib.Path(model_output_dir)
    model_output_dir.mkdir(exist_ok=True)
    model_file = model_output_dir / "state_dict.pth"
    test_loss_fp = model_output_dir / "test_loss.npy"
    test_acc_fp = model_output_dir / "test_acc.npy"

    metadata = {
        "dataset": join(dirname(dirname(abspath(__file__))), dataset),
        "mean": mean,
        "std": std,
        "slm": slm,
        "seed": seed,
        "sensor": sensor,
        "sensor_activation": sensor_act,
        "model": model.name(),
        "model_param": {
            "input_shape": input_shape.tolist(),
            "crop_fact": crop_fact,
            "device": device,
            "deadspace": not simple,
            "scene2mask": scene2mask,
            "mask2sensor": mask2sensor,
            "device_mask_creation": device_mask_creation,
            "output_dim": np.array(output_dim).tolist(),
            "multi_gpu": multi_gpu,
            "dropout": dropout,
        },
        "batch_size": batch_size,
        "noise_type": noise_type,
        "snr": None if noise_type is None else snr,
    }
    metadata_fp = model_output_dir / "metadata.json"
    with open(metadata_fp, "w") as fp:
        json.dump(metadata, fp)

    print("\nStart training...")
    start_time = time.time()
    if cont:
        test_loss = list(np.load(str(cont / "test_loss.npy")))
        test_accuracy = list(np.load(str(cont / "test_acc.npy")))
        best_test_acc = np.max(test_accuracy)
        best_test_acc_epoch = np.argmax(test_accuracy) + 1
    else:
        test_loss = []
        test_accuracy = []
        best_test_acc = 0
        best_test_acc_epoch = 0
    for epoch in range(n_epoch):
        # training
        running_loss = 0.0
        for i, (x, target) in enumerate(train_loader):

            # print(i)
            # mask_old = model.slm_vals.clone()
            # weights_old = model.multiclass_logistic_reg[0].weight.clone()

            # get inputs
            if use_cuda:
                x, target = x.to(device), target.to(device)

            # zero parameters gradients
            optimizer.zero_grad()

            # forward, backward, optimize
            out = model(x)
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()

            # ensure model weights are between [0, 1]
            with torch.no_grad():
                model.slm_vals.clamp_(min=0, max=1)

            # SLM values have updated after backward
            # TODO : move into forward?
            model.compute_intensity_psf()

            # mask_new = model.slm_vals.clone()
            # weights_new = model.multiclass_logistic_reg[0].weight.clone()
            # import pudb; pudb.set_trace()
            # print("mask change", (mask_new - mask_old).sum())
            # (weights_new - weights_old).sum()
            # model.state_dict()

            # print statistics
            running_loss += loss.item()
            if (i + 1) % print_epoch == 0:  # print every `print_epoch` mini-batches
                proc_time = (time.time() - start_time) / 60.0
                print(
                    f"[{epoch + 1}, {i + 1:5d}, {proc_time:.2f} min] loss: {running_loss / print_epoch:.3f}"
                )
                running_loss = 0.0

            # if i % batch_size == (batch_size - 1):  # print every X mini-batches
            #     print(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / batch_size:.3f}")
            #     running_loss = 0.0

        # testing
        correct_cnt, running_loss = 0, 0
        total_cnt = 0
        for i, (x, target) in enumerate(test_loader):

            # get inputs
            if use_cuda:
                x, target = x.to(device), target.to(device)

            # forward, and compute loss
            out = model(x)
            loss = criterion(out, target)
            _, pred_label = torch.max(out.data, 1)
            total_cnt += x.data.size()[0]
            correct_cnt += (pred_label == target.data).sum()
            running_loss += loss.item() / batch_size
        proc_time = (time.time() - start_time) / 60.0
        _acc = (correct_cnt * 1.0 / total_cnt).item()
        print(
            "==>>> epoch: {}, , {:.2f} min, test loss: {:.6f}, acc: {:.3f}".format(
                epoch + 1, proc_time, running_loss, _acc
            )
        )
        test_loss.append(running_loss)
        test_accuracy.append(_acc)

        if _acc > best_test_acc:
            # save model param
            best_test_acc = _acc
            best_test_acc_epoch = epoch + 1
            torch.save(model.state_dict(), str(model_file))

        # save losses
        with open(test_loss_fp, "wb") as f:
            np.save(f, np.array(test_loss))
        with open(test_acc_fp, "wb") as f:
            np.save(f, np.array(test_accuracy))

    proc_time = time.time() - start_time
    print(f"Processing time [m] : {proc_time / 60}")
    print("Finished Training")

    # save model metadata
    metadata.update(
        {
            "best_test_acc": best_test_acc,
            "best_test_acc_epoch": best_test_acc_epoch,
        }
    )
    metadata_fp = model_output_dir / "metadata.json"
    with open(metadata_fp, "w") as fp:
        json.dump(metadata, fp)

    print(f"Model saved to : {str(model_output_dir)}")


if __name__ == "__main__":
    train_slm_logistic_reg()
