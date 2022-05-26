from lenslessclass.models import MultiClassLogistic, SingleHidden, DeepBig
from lenslessclass.datasets import simulate_propagated_dataset
import torch
import torch.nn as nn
from lenslessclass.datasets import MNISTAugmented
import torch.optim as optim
import torchvision.transforms as transforms
import time
import click
import torchvision.datasets as dset
from waveprop.devices import SensorOptions
import pathlib as plib
import numpy as np
import json
from os.path import dirname, abspath, join
import os
import random
from datetime import datetime
from lenslessclass.util import device_checks
from waveprop.devices import SensorOptions, sensor_dict, SensorParam
from pprint import pprint


@click.command()
@click.option(
    "--dataset",
    type=str,
    help="Path to dataset.",
)
@click.option("--lr", type=float, help="Learning rate for SGD.", default=0.01)
@click.option("--momentum", type=float, help="Momentum for SGD.", default=0.01)
@click.option("--n_epoch", type=int, help="Number of epochs to train.", default=10)
@click.option("--seed", type=int, help="Random seed.", default=0)
@click.option("--batch_size", type=int, help="Batch size.", default=100)
@click.option(
    "--opti",
    type=click.Choice(["sgd", "adam"], case_sensitive=False),
    help="Optimizer.",
    default="adam",
)

# parameters for creating dataset, as in `scripts/save_simulated_dataset.py`
@click.option(
    "--psf",
    type=str,
    help="Path to PSF.",
)
@click.option("--output_dir", type=str, default="data", help="Path to save augmented dataset.")
@click.option("--down_out", type=float, help="Factor by which to downsample output.", default=None)
@click.option(
    "--output_dim",
    default=None,
    nargs=2,
    type=int,
    help="Output dimension (height, width). Use this instead of `down_out` if provided",
)
@click.option("--scene2mask", type=float, default=0.4, help="Scene to SLM/mask distance in meters.")
@click.option(
    "--mask2sensor", type=float, default=0.004, help="SLM/mask to sensor distance in meters."
)
@click.option(
    "--down_psf", type=float, help="Factor by which to downsample convolution.", default=2
)
@click.option("--object_height", type=float, default=0.12, help="Object height in meters.")
@click.option(
    "--single_psf",
    is_flag=True,
    help="Same PSF for all channels (sum) or unique PSF for RGB.",
)
@click.option(
    "--rgb",
    is_flag=True,
    help="Whether to do RGB simulation.",
)
@click.option(
    "--crop_output",
    is_flag=True,
    help="Crop output before downsampling (needed for lens PSF if small object height).",
)
@click.option(
    "--crop_psf",
    type=int,
    help="Crop PSF, needed for lens!! Number of pixels along each dimension.",
)
@click.option(
    "--noise_type",
    default=None,
    type=click.Choice(["speckle", "gaussian", "s&p", "poisson"]),
    help="Noise type to add.",
)
@click.option("--snr", default=40, type=float, help="SNR to determine noise to add.")
@click.option("--sensor", type=str, help="Which sensor to use.", default=SensorOptions.RPI_HQ.value)
@click.option("--n_files", type=int, default=None)
@click.option(
    "--hidden", type=int, default=None, help="If defined, add a hidden layer with this many units."
)
@click.option("--device", type=str, help="Main device for training.")
@click.option(
    "--down_orig",
    default=1,
    type=float,
    help="Amount to downsample original.",
)
@click.option(
    "--single_gpu",
    is_flag=True,
    help="Whether to use single GPU is multiple available. Default will try using all.",
)
@click.option(
    "--mean",
    type=float,
    help="Mean of original dataset to normalize, if not provided it will be computed.",
)
@click.option(
    "--std",
    type=float,
    help="Standard deviation of original dataset to normalize, if not provided it will be computed.",
)
@click.option(
    "--cont",
    type=str,
    help="Path to training to continue.",
)
@click.option(
    "--down",
    default="resize",
    type=click.Choice(["resize", "max", "avg"], case_sensitive=False),
    help="Method for downsampling / reducing dimension.",
)
@click.option(
    "--shift",
    is_flag=True,
    help="Whether to random shift object in scene.",
)
@click.option(
    "--deepbig",
    is_flag=True,
    help="Whether to use deep big model.",
)
@click.option(
    "--random_height",
    default=None,
    nargs=2,
    type=float,
    help="Random height range in cm.",
)
@click.option(
    "--rotate",
    default=False,
    type=float,
    help="Random degrees to rotate: (-rotate, rotate).",
)
@click.option(
    "--perspective",
    default=False,
    type=float,
    help="Distortion scale for random perspective.",
)
@click.option(
    "--dropout",
    default=None,
    type=float,
    help="Percentage of dropout after each layer in deep model.",
)
def train_fixed_encoder(
    dataset,
    seed,
    lr,
    momentum,
    n_epoch,
    batch_size,
    opti,
    psf,
    output_dir,
    down_out,
    scene2mask,
    mask2sensor,
    down_psf,
    object_height,
    single_psf,
    rgb,
    crop_output,
    sensor,
    n_files,
    output_dim,
    crop_psf,
    noise_type,
    snr,
    hidden,
    device,
    down_orig,
    single_gpu,
    mean,
    std,
    cont,
    down,
    shift,
    random_height,
    deepbig,
    dropout,
    rotate,
    perspective,
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
        seed = metadata["seed"]
        multi_gpu = metadata["model_param"]["multi_gpu"]
        single_gpu = metadata["single_gpu"]
        batch_size = metadata["batch_size"]

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    if crop_psf:
        down_psf = 1

    device, use_cuda, multi_gpu, device_ids = device_checks(device=device, single_gpu=single_gpu)

    target_dim = np.array([28, 28])
    if down_orig:
        w = int(np.round(np.sqrt(np.prod(target_dim) / down_orig)))
        target_dim = np.array([w, w])
        print(f"New target dimension : {target_dim}")
        print(f"Flattened : {w * w}")

    ## load mnist dataset
    if dataset is None and psf is None:

        # w = int(np.sqrt(784 / down_fact * 3040 / 4056))
        # h = int(4056 / 3040 * w)
        # output_dim_mnist = (w, h)
        # print(output_dim_mnist)
        # print(w * h)

        # use original dataset
        print("\nNo dataset nor PSF provided, using original MNIST dataset!\n")

        root = "./data"
        if not os.path.exists(root):
            os.mkdir(root)
        mean = 0.5
        std = 1.0
        trans_list = [transforms.ToTensor(), transforms.Normalize((mean,), (std,))]
        if down_orig:
            trans_list.append(transforms.Resize(size=target_dim.tolist()))
        trans = transforms.Compose(trans_list)
        train_set = dset.MNIST(root=root, train=True, transform=trans, download=True)
        test_set = dset.MNIST(root=root, train=False, transform=trans, download=True)

        output_dim = np.array(list(train_set[0][0].shape))

    if psf:

        if "lens" in psf:
            assert crop_psf is not None

        if random_height is not None:
            random_height = np.array(random_height) * 1e-2
            object_height = None

        sensor_param = sensor_dict[sensor]
        sensor_size = sensor_param[SensorParam.SHAPE]
        if down_out:
            output_dim = (sensor_size * 1 / down_out).astype(int)
        elif down_orig:
            # determine output dim so that sensor measurement is
            # scaled so that aspect ratio is preserved
            n_hidden = np.prod(target_dim)
            w = int(np.sqrt(n_hidden * sensor_size[0] / sensor_size[1]))
            h = int(sensor_size[1] / sensor_size[0] * w)
            output_dim = (w, h)

        print(f"Output dimension : {output_dim}")
        print(f"Downsampling factor : {sensor_size[1] / output_dim[1]}")

        # generate dataset from provided PSF
        args = {
            "dataset": "MNIST",
            "psf": psf,
            "sensor": sensor,
            "output_dir": output_dir,
            "down_psf": down_psf,
            "output_dim": output_dim,
            "scene2mask": scene2mask,
            "mask2sensor": mask2sensor,
            "object_height": object_height,
            "device": device,
            "device_conv": device,
            "crop_output": crop_output,
            "grayscale": not rgb,
            "single_psf": single_psf,
            "n_files": n_files,
            "crop_psf": crop_psf,
            "noise_type": noise_type,
            "snr": snr,
            "down": down,
            "batch_size": batch_size,
            "random_shift": shift,
            "random_height": random_height,
            "rotate": rotate,
            "perspective": perspective,
        }
        dataset = simulate_propagated_dataset(**args)
        print()

    if dataset:

        # -- determine mean and standard deviation (of training set)
        if mean is None and std is None:
            trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0, 1)])
            train_set = MNISTAugmented(path=dataset, train=True, transform=trans)
            mean, std = train_set.get_stats()
            print("Dataset mean : ", mean)
            print("Dataset standard deviation : ", std)

        # -- normalize according to training set stats
        trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        train_set = MNISTAugmented(path=dataset, train=True, transform=trans)
        test_set = MNISTAugmented(path=dataset, train=False, transform=trans)
        output_dim = train_set.output_dim

    train_loader = torch.utils.data.DataLoader(
        dataset=train_set, batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set, batch_size=batch_size, shuffle=False
    )

    print("==>>> total training batch number: {}".format(len(train_loader)))
    print("==>>> total testing batch number: {}".format(len(test_loader)))

    ## training
    if deepbig:
        model = DeepBig(input_shape=output_dim, n_class=10, dropout=dropout)
        model_name = model.name()
        if multi_gpu:
            model = nn.DataParallel(model, device_ids=device_ids)
    elif hidden:
        model = SingleHidden(input_shape=output_dim, hidden_dim=hidden, n_class=10)
        model_name = model.name()
        if multi_gpu:
            model = nn.DataParallel(model, device_ids=device_ids)
    else:
        model = MultiClassLogistic(input_shape=output_dim, multi_gpu=device_ids)
        model_name = model.name()
    if use_cuda:
        model = model.to(device)

    if cont:
        state_dict_fp = str(cont / "state_dict.pth")
        model.load_state_dict(torch.load(state_dict_fp))

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

    print("\nModel parameters:")
    for name, params in model.named_parameters():
        print(name, "\t", params.size(), "\t", params.requires_grad)
    print()
    print("Optimizer's state_dict:")
    for var_name in optimizer.state_dict():
        print(var_name, "\t", optimizer.state_dict()[var_name])

    ## save best model param
    timestamp = datetime.now().strftime("%d%m%Y_%Hh%M")
    if dataset is None:
        model_output_dir = f"./MNIST_original_{int(np.prod(output_dim))}dim_{n_epoch}epoch_seed{seed}_{model_name}_{timestamp}"
    else:
        model_output_dir = (
            f"./{os.path.basename(dataset)}_{n_epoch}epoch_seed{seed}_{model_name}_{timestamp}"
        )
    model_output_dir = plib.Path(model_output_dir)
    model_output_dir.mkdir(exist_ok=True)
    model_file = model_output_dir / "state_dict.pth"

    metadata = {
        "dataset": join(dirname(dirname(abspath(__file__))), dataset)
        if dataset is not None
        else None,
        "seed": seed,
        "mean": mean,
        "std": std,
        "timestamp (DDMMYYYY_HhM)": timestamp,
        "model": model_name,
        "model_param": {"input_shape": output_dim.tolist(), "multi_gpu": device_ids},
        "batch_size": batch_size,
        "hidden_dim": hidden,
        "deepbig": deepbig,
        "noise_type": noise_type,
        "snr": None if noise_type is None else snr,
        "dropout": dropout,
    }
    metadata_fp = model_output_dir / "metadata.json"
    with open(metadata_fp, "w") as fp:
        json.dump(metadata, fp)

    test_loss_fp = model_output_dir / "test_loss.npy"
    test_acc_fp = model_output_dir / "test_acc.npy"
    train_acc_fp = model_output_dir / "train_acc.npy"

    print(f"Model saved to : {str(model_output_dir)}")

    print("Start training...")
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
                x, target = x.to(device), target.to(device)

            # zero parameters gradients
            optimizer.zero_grad()

            # forward, backward, optimize
            out = model(x)
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()

            # train accuracy
            _, pred_label = torch.max(out.data, 1)
            total_cnt += x.data.size()[0]
            correct_cnt += (pred_label == target.data).sum()

            # print statistics
            running_loss += loss.item()
            if i % batch_size == (batch_size - 1):  # print every X mini-batches
                print(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / batch_size:.3f}")
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
                x, target = x.to(device), target.to(device)

            # forward, and compute loss
            out = model(x)
            loss = criterion(out, target)
            _, pred_label = torch.max(out.data, 1)
            total_cnt += x.data.size()[0]
            correct_cnt += (pred_label == target.data).sum()
            running_loss += loss.item() / batch_size
        _loss = running_loss
        _acc = (correct_cnt * 1.0 / total_cnt).item()
        print("==>>> epoch: {}, test loss: {:.6f}, acc: {:.3f}".format(epoch + 1, _loss, _acc))
        test_loss.append(_loss)
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
            np.save(f, np.array(train_acc))

    proc_time = time.time() - start_time
    print(f"Processing time [m] : {proc_time / 60}")
    print("Finished Training")

    ## save model metadata
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
    train_fixed_encoder()
