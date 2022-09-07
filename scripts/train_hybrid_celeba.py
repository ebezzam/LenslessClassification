"""

Train SLM as well as digital model

"""

from lenslessclass.models import SLMMultiClassLogistic
import torch
import random
from pprint import pprint
import pathlib as plib
import os
import torch.nn as nn
import pandas as pd
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
from lenslessclass.datasets import (
    CelebAAugmented,
    CELEBA_ATTR,
    simulate_propagated_dataset,
    get_dataset_stats,
)
from torch.utils.data import Subset
from lenslessclass.util import device_checks
from sklearn.model_selection import train_test_split
import torchvision.datasets as dset


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
    "--output_dir",
    type=str,
    default="data_celeba",
    help="Path to save augmented dataset (if created).",
)
@click.option("--simple", is_flag=True, help="Don't take into account deadspace.")
@click.option(
    "--scene2mask", type=float, default=0.55, help="Scene to SLM/mask distance in meters."
)
@click.option(
    "--mask2sensor", type=float, default=0.004, help="SLM/mask to sensor distance in meters."
)
@click.option("--object_height", type=float, default=0.27, help="Object height in meters.")
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
    "--down",
    default="resize",
    type=click.Choice(["resize", "max", "avg"], case_sensitive=False),
    help="Method for downsampling / reducing dimension.",
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
@click.option(
    "--root",
    type=str,
    default="/scratch",
    help="Parent directory of `celeba`.",
)
@click.option(
    "--down_orig",
    type=float,
    help="Amount to downsample original.",
)
@click.option(
    "--attr",
    type=click.Choice(CELEBA_ATTR, case_sensitive=True),
    help="Attribute to predict.",
)
@click.option(
    "--test_size",
    type=float,
    default=0.15,
    help="Test size ratio.",
)
@click.option(
    "--hidden", type=int, default=None, help="If defined, add a hidden layer with this many units."
)
@click.option(
    "--min_val",
    type=float,
    default=None,
    help="Minimum value from train set to make images non-negative. Default is to determine.",
)
@click.option(
    "--use_max_range",
    is_flag=True,
    help="Normalize simulated data to maximum bit depth. Otherwise random but no clipping.",
)
@click.option(
    "--cnn",
    default=None,
    type=int,
    help="Use CNN model as classifier. Argument denotes number of kernels.",
)
@click.option(
    "--n_mask",
    default=1,
    type=int,
    help="Number of SLM masks to optimized, that would be applied in time-multiplexed fashion.",
)
@click.option(
    "--cnn_lite",
    default=None,
    type=int,
    help="Use CNNLite model as classifier. Argument denotes number of kernels.",
)
@click.option(
    "--pool",
    default=2,
    type=int,
    help="Pooling for CNN models.",
)
@click.option(
    "--kernel_size",
    default=3,
    type=int,
    help="Kernel size for CNN models.",
)
def train_hybrid_celeba(
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
    root,
    down_orig,
    attr,
    test_size,
    hidden,
    min_val,
    use_max_range,
    down,
    cnn,
    n_mask,
    cnn_lite,
    pool,
    kernel_size,
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
        if "down_orig" in metadata.keys():
            down_orig = metadata["down_orig"]
        if "hidden" in metadata["model_param"].keys():
            hidden = metadata["model_param"]["hidden"]
        if "min_val" in metadata.keys():
            min_val = metadata["min_val"]
        if "n_kern" in metadata.keys():
            cnn = metadata["n_kern"]
        if "n_slm_mask" in metadata.keys():
            n_mask = metadata["n_slm_mask"]

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    device, use_cuda, multi_gpu, device_ids = device_checks(device=device, single_gpu=single_gpu)
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

    ## load dataset
    # -- load original to have same split
    trans_list = [
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (1.0,)),
        transforms.Grayscale(num_output_channels=1),
    ]

    target_dim = np.array([218, 178])
    if down_orig:
        target_dim = (target_dim / down_orig).astype(int)
        trans_list.append(transforms.Resize(size=target_dim.tolist()))

    trans = transforms.Compose(trans_list)
    # -- TODO can avoid loading dataset as we know number of files and order of attributes
    ds = dset.CelebA(
        root=root,
        split="all",
        download=False,
        transform=trans,
    )
    if n_files is None:
        n_files = len(ds)
        train_size = 1 - test_size
    else:
        print(f"Using {n_files}")
        test_size = int(n_files * test_size)
        train_size = n_files - test_size
    label_idx = ds.attr_names.index(attr)
    labels = ds.attr[:, label_idx][:n_files]
    train_indices, test_indices, _, _ = train_test_split(
        range(n_files),
        labels,
        train_size=train_size,
        test_size=test_size,
        stratify=labels,
        random_state=seed,
    )

    print(f"\ntrain set - {len(train_indices)}")
    df_attr = pd.DataFrame(ds.attr[train_indices, label_idx])
    print(df_attr.value_counts() / len(df_attr))

    print(f"\ntest set - {len(test_indices)}")
    df_attr = pd.DataFrame(ds.attr[test_indices, label_idx])
    print(df_attr.value_counts() / len(df_attr))

    # determine output dim so that sensor measurement is
    # scaled so that aspect ratio is preserved
    sensor_param = sensor_dict[sensor]
    sensor_size = sensor_param[SensorParam.SHAPE]

    if output_dim is None:
        if down_out:
            output_dim = tuple((sensor_size / down_out).astype(int))
        elif down_orig:
            # determine output dim so that sensor measurement is
            # scaled so that aspect ratio is preserved
            n_hidden = np.prod(target_dim)
            w = int(np.sqrt(n_hidden * sensor_size[0] / sensor_size[1]))
            h = int(sensor_size[1] / sensor_size[0] * w)
            output_dim = (w, h)

    print(f"Output dimension : {output_dim}")
    print(f"Downsampling factor : {sensor_size[1] / output_dim[1]}")

    # check if need to create dataset
    if dataset is None:
        dataset = simulate_propagated_dataset(
            dataset="celeba",
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
            use_max_range=use_max_range,
            # don't need noise parameters as done on the fly
        )

    # -- first determine mean and standard deviation (of training set)
    if mean is None and std is None:
        trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0, 1)])
        print("\nComputing stats...")

        all_data = CelebAAugmented(path=dataset, transform=trans)
        train_set = Subset(all_data, train_indices)
        mean, std = get_dataset_stats(train_set)
        print("Dataset mean : ", mean)
        print("Dataset standard deviation : ", std)

        del all_data

    # -- normalize according to training set stats
    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    all_data = CelebAAugmented(path=dataset, transform=trans)
    train_set = Subset(all_data, train_indices)
    test_set = Subset(all_data, test_indices)
    input_shape = np.array(list(train_set[0][0].squeeze().shape))

    train_loader = torch.utils.data.DataLoader(
        dataset=train_set, batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set, batch_size=batch_size, shuffle=False
    )

    print(f"number training examples: {len(train_set)}")
    print(f"number test examples: {len(test_set)}")
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
        down=down,
        sensor_activation=sensor_act_fn,
        multi_gpu=device_ids,
        dropout=dropout,
        noise_type=noise_type,
        snr=snr,
        # target_dim=target_dim,
        n_class=1,
        hidden=hidden,
        n_kern=cnn,
        n_slm_mask=n_mask,
        pool=pool,
        kernel_size=kernel_size,
        cnn_lite=cnn_lite,
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

    criterion = nn.BCELoss()

    # Print model and optimizer state_dict
    print("\nModel's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    print()
    print("Optimizer's state_dict:")
    for var_name in optimizer.state_dict():
        print(var_name, "\t", optimizer.state_dict()[var_name])

    if min_val is None:
        min_val = 0
        for (x, target) in train_loader:
            if x.min() < min_val:
                min_val = x.min()
        min_val = float(min_val.item())
        print("Minimum value : ", min_val)

    ## save best model param
    n_hidden = np.prod(output_dim)
    timestamp = datetime.now().strftime("%d%m%Y_%Hh%M")
    model_output_dir = f"./{os.path.basename(dataset)}_scene2mask{scene2mask}_outdim{int(n_hidden)}_{attr}_{n_epoch}epoch_seed{seed}_{model.name()}"
    if noise_type:
        model_output_dir += f"_{noise_type}{snr}"
    model_output_dir += f"_{timestamp}"

    model_output_dir = plib.Path(model_output_dir)
    model_output_dir.mkdir(exist_ok=True)
    model_file = model_output_dir / "state_dict.pth"
    test_loss_fp = model_output_dir / "test_loss.npy"
    test_acc_fp = model_output_dir / "test_acc.npy"
    train_acc_fp = model_output_dir / "train_acc.npy"

    metadata = {
        "dataset": join(dirname(dirname(abspath(__file__))), dataset),
        "down_orig": down_orig,
        "attr": attr,
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
            "down": down,
            "multi_gpu": multi_gpu,
            "dropout": dropout,
            "hidden": hidden,
            "n_kern": cnn,
            "n_slm_mask": n_mask,
            "pool": pool,
            "cnn_lite": cnn_lite,
            "kernel_size": kernel_size,
        },
        "batch_size": batch_size,
        "noise_type": noise_type,
        "snr": None if noise_type is None else snr,
        "device_ids": device_ids,
        "min_val": min_val,
    }
    metadata_fp = model_output_dir / "metadata.json"
    with open(metadata_fp, "w") as fp:
        json.dump(metadata, fp)

    print(f"Model saved to : {str(model_output_dir)}")

    print("\nStart training...")
    start_time = time.time()
    if cont:
        test_loss = list(np.load(str(cont / "test_loss.npy")))
        test_accuracy = list(np.load(str(cont / "test_acc.npy")))
        train_accuracy = list(np.load(str(cont / "train_acc.npy")))
        best_test_acc = np.max(test_accuracy)
        best_test_acc_epoch = np.argmax(test_accuracy) + 1
    else:
        test_loss = []
        test_accuracy = []
        train_accuracy = []
        best_test_acc = 0
        best_test_acc_epoch = 0
    for epoch in range(n_epoch):
        # training
        running_loss = 0.0
        correct_cnt, total_cnt = 0, 0
        for i, (x, target) in enumerate(train_loader):

            # get inputs
            if use_cuda:
                x = x.to(device)
            target = target[:, label_idx]
            target = target.unsqueeze(1)
            target = target.to(x)

            # non-negative
            x -= min_val

            # zero parameters gradients
            optimizer.zero_grad()

            # forward, backward, optimize
            out = model(x)
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()

            # train accuracy
            pred_label = out.round()
            correct_cnt += (pred_label == target).sum()
            total_cnt += x.data.size()[0]

            # ensure model weights are between [0, 1]
            with torch.no_grad():
                if n_mask == 1:
                    model.slm_vals.clamp_(min=0, max=1)
                else:
                    [model.slm_vals[i].clamp_(min=0, max=1) for i in range(model.n_slm_mask)]

                # model.slm_vals *= 255
                # model.slm_vals = torch.nn.Parameter(
                #     model.slm_vals.to(dtype=torch.uint8).to(dtype=torch.float32) / 255
                # )

            # SLM values have updated after backward
            # TODO : move into forward?
            model.compute_intensity_psf()

            # print statistics
            running_loss += loss.item() / batch_size
            if (i + 1) % print_epoch == 0:  # print every `print_epoch` mini-batches
                proc_time = (time.time() - start_time) / 60.0
                print(f"[{epoch + 1}, {i + 1:5d}, {proc_time:.2f} min] loss: {running_loss:.3f}")
                running_loss = 0.0

        train_acc = (correct_cnt * 1.0 / total_cnt).item()
        train_accuracy.append(train_acc)
        print(f"training accuracy : {train_acc:.3f}")

        # testing
        correct_cnt, running_loss = 0, 0
        total_cnt = 0
        for i, (x, target) in enumerate(test_loader):

            # get inputs
            if use_cuda:
                x = x.to(device)
            target = target[:, label_idx]
            target = target.unsqueeze(1)
            target = target.to(x)

            # non-negative
            x -= min_val

            # forward, and compute loss
            out = model(x)
            loss = criterion(out, target)

            # compute accuracy
            pred_label = out.round()
            correct_cnt += (pred_label == target).sum()
            total_cnt += x.data.size()[0]
            running_loss += loss.item() / batch_size

        proc_time = (time.time() - start_time) / 60.0
        _acc = (correct_cnt * 1.0 / total_cnt).item()
        print(
            "==>>> epoch: {}, {:.2f} min, test loss: {:.6f}, acc: {:.3f}".format(
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
        with open(train_acc_fp, "wb") as f:
            np.save(f, np.array(train_accuracy))

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
    train_hybrid_celeba()
