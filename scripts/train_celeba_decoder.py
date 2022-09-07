from lenslessclass.models import BinaryLogistic, TwoHidden, SingleHidden, CNN, CNNLite
import torch
import torch.nn as nn
from lenslessclass.datasets import (
    CelebAAugmented,
    CELEBA_ATTR,
    simulate_propagated_dataset,
    get_dataset_stats,
)
import torch.optim as optim
import torchvision.transforms as transforms
import time
import click
import torchvision.datasets as dset
from torch.utils.data import Subset
from waveprop.devices import SensorOptions, sensor_dict, SensorParam
import pathlib as plib
import numpy as np
import json
from os.path import dirname, abspath, join
import os
import random
from datetime import datetime
from sklearn.model_selection import train_test_split
from pprint import pprint
from lenslessclass.util import device_checks
import pandas as pd
from lenslessclass.generator import SingleHidden


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

# original dataset
@click.option(
    "--root",
    type=str,
    default="/scratch",
    help="Parent directory of `celeba`.",
)
@click.option(
    "--test_size",
    type=float,
    default=0.15,
    help="Test size ratio.",
)
@click.option(
    "--attr",
    type=click.Choice(CELEBA_ATTR, case_sensitive=True),
    help="Attribute to predict.",
)

# parameters for creating dataset
@click.option(
    "--psf",
    type=str,
    help="Path to PSF.",
)
@click.option(
    "--output_dir", type=str, default="data_celeba", help="Path to save augmented dataset."
)
@click.option("--down_out", type=float, help="Factor by which to downsample output.", default=None)
@click.option(
    "--scene2mask", type=float, default=0.55, help="Scene to SLM/mask distance in meters."
)
@click.option(
    "--mask2sensor", type=float, default=0.004, help="SLM/mask to sensor distance in meters."
)
@click.option(
    "--down_psf", type=float, help="Factor by which to downsample convolution.", default=1
)
@click.option("--object_height", type=float, default=0.27, help="Object height in meters.")
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
@click.option(
    "--use_max_range",
    is_flag=True,
    help="Normalize simulated data to maximum bit depth. Otherwise random but no clipping.",
)
@click.option(
    "--down_orig",
    type=float,
    help="Amount to downsample original.",
)
@click.option(
    "--single_gpu",
    is_flag=True,
    help="Whether to use single GPU is multiple available. Default will try using all.",
)
@click.option("--n_files", type=int, default=None, help="For testing purposes.")
@click.option(
    "--output_dim",
    default=None,
    nargs=2,
    type=int,
    help="Output dimension (height, width).",
)
@click.option("--hidden", type=int, default=10000, help="Hidden layer with this many units.")
@click.option("--device", type=str, help="Main device for training.")
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
def train_decoder(
    root,
    attr,
    test_size,
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
    use_max_range,
    mean,
    std,
):
    if n_files == 0:
        n_files = None

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    if crop_psf:
        down_psf = 1

    device, use_cuda, multi_gpu, device_ids = device_checks(device=device, single_gpu=single_gpu)

    ## LOAD DATASET

    # -- load original to have same split
    trans_list = [
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (1,)),
        transforms.Grayscale(num_output_channels=1),
    ]

    target_dim = np.array([218, 178])
    if down_orig:
        target_dim = (target_dim / down_orig).astype(int)
        trans_list.append(transforms.Resize(size=target_dim.tolist()))

    trans = transforms.Compose(trans_list)
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

    # -- simulate
    if "lens" in psf:
        assert crop_psf is not None

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

    dataset = simulate_propagated_dataset(
        dataset="celeba",
        psf=psf,
        sensor=sensor,
        output_dir=output_dir,
        down_psf=down_psf,
        output_dim=output_dim,
        scene2mask=scene2mask,
        mask2sensor=mask2sensor,
        object_height=object_height,
        crop_output=crop_output,
        grayscale=not rgb,
        single_psf=single_psf,
        n_files=n_files,
        crop_psf=crop_psf,
        noise_type=noise_type,
        snr=snr,
        batch_size=batch_size,
        device_conv=device,
        use_max_range=use_max_range,
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
    all_data = CelebAAugmented(path=dataset, transform=trans, return_original=root)
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

    # load model
    model = SingleHidden(input_shape=output_dim, hidden_dim=hidden, n_output=np.prod(target_dim))
    model_name = model.name()
    if multi_gpu:
        model = nn.DataParallel(model, device_ids=device_ids)
    if use_cuda:
        model = model.to(device)

    # set optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=0.001,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=0,
        amsgrad=False,
    )

    criterion = nn.MSELoss()

    print("\nModel parameters:")
    for name, params in model.named_parameters():
        print(name, "\t", params.size(), "\t", params.requires_grad)
    print()
    print("Optimizer's state_dict:")
    for var_name in optimizer.state_dict():
        print(var_name, "\t", optimizer.state_dict()[var_name])

    ## save best model param
    timestamp = datetime.now().strftime("%d%m%Y_%Hh%M")
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
        "batch_size": batch_size,
        "hidden_dim": hidden,
        "noise_type": noise_type,
        "snr": None if noise_type is None else snr,
    }
    metadata_fp = model_output_dir / "metadata.json"
    with open(metadata_fp, "w") as fp:
        json.dump(metadata, fp)

    test_loss_fp = model_output_dir / "test_loss.npy"
    train_loss_fp = model_output_dir / "train_loss.npy"

    print(f"Model saved to : {str(model_output_dir)}")

    print("Start training...")
    start_time = time.time()
    test_loss = []
    train_loss = []
    best_test_loss = np.inf
    best_test_loss_epoch = 0
    for epoch in range(n_epoch):

        # training
        running_loss = 0.0

        for i, (x, target, x_orig) in enumerate(train_loader):

            # get inputs
            if use_cuda:
                x, target, x_orig = x.to(device), target.to(device), x_orig.to(device)
            x_orig = x_orig.view(-1, np.prod(x_orig.size()[1:]))

            # zero parameters gradients
            optimizer.zero_grad()

            # forward, backward, optimize
            out = model(x)
            loss = criterion(out, x_orig)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item() / batch_size
            if i % batch_size == (batch_size - 1):  # print every X mini-batches
                print(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss:.3f}")

        train_loss.append(running_loss)

        # testing
        running_loss = 0.0
        for i, (x, target, x_orig) in enumerate(test_loader):

            # get inputs
            if use_cuda:
                x, target, x_orig = x.to(device), target.to(device), x_orig.to(device)
            x_orig = x_orig.view(-1, np.prod(x_orig.size()[1:]))

            # forward, and compute loss
            out = model(x)
            loss = criterion(out, x_orig)

            running_loss += loss.item() / batch_size

        print("==>>> epoch: {}, test loss: {:.6f}".format(epoch + 1, running_loss))
        test_loss.append(running_loss)

        if running_loss < best_test_loss:
            # save model param
            best_test_loss = running_loss
            best_test_loss_epoch = epoch + 1
            torch.save(model.state_dict(), str(model_file))

        # save losses
        with open(test_loss_fp, "wb") as f:
            np.save(f, np.array(test_loss))
        with open(train_loss_fp, "wb") as f:
            np.save(f, np.array(train_loss))

    proc_time = time.time() - start_time
    print(f"Processing time [m] : {proc_time / 60}")
    print("Finished Training")

    ## save model metadata
    metadata.update(
        {
            "best_test_loss": best_test_loss,
            "best_test_loss_epoch": best_test_loss_epoch,
        }
    )
    metadata_fp = model_output_dir / "metadata.json"
    with open(metadata_fp, "w") as fp:
        json.dump(metadata, fp)

    print(f"Model saved to : {str(model_output_dir)}")


if __name__ == "__main__":
    train_decoder()
