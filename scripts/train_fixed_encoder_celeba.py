from lenslessclass.models import BinaryLogistic, MultiClassLogistic, SingleHidden
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
    default=0.1,
    help="Test size ratio.",
)
@click.option(
    "--single_gpu",
    is_flag=True,
    help="Whether to use single GPU is multiple available. Default will try using all.",
)
@click.option(
    "--cont",
    type=str,
    help="Path to training to continue.",
)
@click.option("--n_files", type=int, default=None, help="For testing purposes.")
@click.option(
    "--psf",
    type=str,
    help="Path to PSF.",
)
@click.option("--output_dir", type=str, default="data", help="Path to save augmented dataset.")
@click.option(
    "--output_dim",
    default=None,
    nargs=2,
    type=int,
    help="Output dimension (height, width).",
)
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
    "--hidden", type=int, default=None, help="If defined, add a hidden layer with this many units."
)
@click.option("--device", type=str, help="Main device for training.")
def train_fixed_encoder(
    down_orig,
    attr,
    test_size,
    device,
    single_gpu,
    dataset,
    seed,
    lr,
    momentum,
    n_epoch,
    batch_size,
    opti,
    psf,
    output_dir,
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
    cont,
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
        down_orig = metadata["down_orig"]
        multi_gpu = metadata["model_param"]["multi_gpu"]
        single_gpu = metadata["single_gpu"]
        batch_size = metadata["batch_size"]
        noise_type = metadata["noise_type"]
        snr = metadata["snr"]
        attr = metadata["attr"]
        crop_psf = metadata["crop_psf"]
        output_dim = metadata["output_dim"]

        input_shape = np.array(metadata["model_param"]["input_shape"])
        hidden_dim = np.array(metadata["hidden_dim"])

    assert attr is not None
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

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

    ## LOAD DATASET

    # -- load original to have same split
    root = "./data"
    mean = 0.5
    std = 1.0
    trans_list = [
        transforms.ToTensor(),
        transforms.Normalize((mean,), (std,)),
        transforms.Grayscale(num_output_channels=1),
    ]

    target_dim = (218, 178)
    if down_orig:
        orig_dim = np.array(target_dim)
        target_dim = tuple((orig_dim / down_orig).astype(int))
        trans_list.append(transforms.Resize(size=target_dim))

    trans = transforms.Compose(trans_list)
    ds = dset.CelebA(
        root=root,
        split="all",
        download=False,
        transform=trans,
    )
    if n_files is None:
        n_files = len(ds)
    else:
        print(f"TEST : using {n_files}")
    label_idx = ds.attr_names.index(attr)
    labels = ds.attr[:, label_idx][:n_files]
    train_indices, test_indices, _, _ = train_test_split(
        range(n_files), labels, test_size=test_size, stratify=labels, random_state=seed
    )

    if dataset is None and psf is None:

        # use original dataset
        print("\nNo dataset nor PSF provided, using original CelebA dataset!\n")
        noise_type = None

        train_set = Subset(ds, train_indices)
        test_set = Subset(ds, test_indices)
        output_dim = np.array(list(train_set[0][0].shape))

    if psf:

        if "lens" in psf:
            assert crop_psf is not None

        # determine output dim so that sensor measurement is
        # scaled so that aspect ratio is preserved
        sensor_param = sensor_dict[sensor]
        sensor_size = sensor_param[SensorParam.SHAPE]
        n_hidden = np.prod(target_dim)
        w = np.sqrt(n_hidden * sensor_size[1] / sensor_size[0])
        h = sensor_size[0] / sensor_size[1] * w
        output_dim = (int(np.round(h)), int(np.round(w)))
        print(f"Output dimension : {output_dim}")
        print(f"Downsampling factor : {sensor_size[1] / w}")

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
            n_workers=0,
        )
        print()

    if dataset:

        # -- determine mean and standard deviation (of training set)
        trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0, 1)])
        all_data = CelebAAugmented(path=dataset, transform=trans)
        train_set = Subset(all_data, train_indices)
        mean, std = get_dataset_stats(train_set)
        print("Dataset mean : ", mean)
        print("Dataset standard deviation : ", std)

        # -- normalize according to training set stats
        trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        all_data = CelebAAugmented(path=dataset, transform=trans)
        train_set = Subset(all_data, train_indices)
        test_set = Subset(all_data, test_indices)
        output_dim = np.array(list(train_set[0][0].shape))

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

    ## training
    if hidden:
        raise NotImplementedError
    else:
        model = BinaryLogistic(input_shape=output_dim, multi_gpu=multi_gpu)
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

    criterion = nn.BCELoss()

    # Print model and optimizer state_dict
    print("\nModel's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    print()
    print("Optimizer's state_dict:")
    for var_name in optimizer.state_dict():
        print(var_name, "\t", optimizer.state_dict()[var_name])

    ## save best model param
    timestamp = datetime.now().strftime("%d%m%Y_%Hh%M")
    if dataset is None:
        model_output_dir = f"./CelebA_original_{n_epoch}epoch_seed{seed}_{model.name()}_{timestamp}"
    else:
        model_output_dir = (
            f"./{os.path.basename(dataset)}_{n_epoch}epoch_seed{seed}_{model.name()}_{timestamp}"
        )
    model_output_dir = plib.Path(model_output_dir)
    model_output_dir.mkdir(exist_ok=True)
    model_file = model_output_dir / "state_dict.pth"

    metadata = {
        "dataset": join(dirname(dirname(abspath(__file__))), dataset)
        if dataset is not None
        else None,
        "down_orig": down_orig,
        "attr": attr,
        "seed": seed,
        "mean": mean,
        "std": std,
        "timestamp (DDMMYYYY_HhM)": timestamp,
        "model": model.name(),
        "model_param": {"input_shape": output_dim.tolist(), "multi_gpu": multi_gpu},
        "batch_size": batch_size,
        "hidden_dim": hidden,
        "noise_type": noise_type,
        "snr": None if noise_type is None else snr,
        "single_gpu": single_gpu,
        "crop_psf": crop_psf,
        "output_dim": output_dim.tolist(),
    }
    metadata_fp = model_output_dir / "metadata.json"
    with open(metadata_fp, "w") as fp:
        json.dump(metadata, fp)

    test_loss_fp = model_output_dir / "test_loss.npy"
    test_acc_fp = model_output_dir / "test_acc.npy"

    print("Start training...")
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
            # get inputs
            if use_cuda:
                x = x.to(device)
            target = target[:, label_idx]
            target = target.unsqueeze(1)
            target = target.to(x)

            # zero parameters gradients
            optimizer.zero_grad()

            # forward, backward, optimize
            out = model(x)

            loss = criterion(out, target)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % batch_size == (batch_size - 1):  # print every X mini-batches
                print(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / batch_size:.3f}")
                running_loss = 0.0

        # testing
        correct_cnt, running_loss = 0, 0
        total_cnt = 0
        for i, (x, target) in enumerate(test_loader):

            # get inputs
            if use_cuda:
                x, target = x.to(device), target[:, label_idx]
            target = target.unsqueeze(1)
            target = target.to(x)

            # forward, and compute loss
            out = model(x)
            loss = criterion(out, target)

            # compute accuracy
            pred_label = out.round()
            correct_cnt += (pred_label == target).sum()
            total_cnt += x.data.size()[0]
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
