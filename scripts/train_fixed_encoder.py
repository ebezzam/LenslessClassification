"""
Modified from this example:
https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

TODO : specify dataset creation, and create if not available!!
TODO : save models


Original MNIST
```
python scripts/train_logistic_reg.py
```

Lens
```
python scripts/train_logistic_reg.py --dataset data/MNIST_lens_down128
```

Tape
```
python scripts/train_logistic_reg.py --dataset data/MNIST_tape_down128
```

SLM
```
python scripts/train_logistic_reg.py --dataset data/MNIST_adafruit_down128
```


"""

from lenslessclass.models import MultiClassLogistic, SingleHidden
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
    default="sgd",
)
@click.option("--cpu", is_flag=True, help="Use CPU even if GPU if available.")

# parameters for creating dataset, as in `scripts/save_simulated_dataset.py`
@click.option(
    "--psf",
    type=str,
    help="Path to PSF.",
)
@click.option("--output_dir", type=str, default="data", help="Path to save augmented dataset.")
@click.option("--down_out", type=float, help="Factor by which to downsample output.", default=128)
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
@click.option(
    "--object_height", type=float, default=5e-2, help="SLM/mask to sensor distance in meters."
)
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
def train_fixed_encoder(
    dataset,
    seed,
    lr,
    momentum,
    n_epoch,
    batch_size,
    opti,
    cpu,
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
):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    use_cuda = torch.cuda.is_available()
    if cpu:
        device = "cpu"
        use_cuda = False
    else:
        if use_cuda:
            print("CUDA available, using GPU.")
            device = "cuda"
            n_gpus = torch.cuda.device_count()
            if n_gpus > 1:
                multi_gpu = True
                print(f"-- using {n_gpus} GPUs")
            else:
                multi_gpu = False

        else:
            device = "cpu"
            print("CUDA not available, using CPU.")

    ## load mnist dataset
    if dataset is None and psf is None:

        # use original dataset
        print("\nNo dataset nor PSF provided, using original MNIST dataset!\n")

        root = "./data"
        if not os.path.exists(root):
            os.mkdir(root)
        mean = 0.5
        std = 1.0
        trans = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((mean,), (std,))]
        )  # need to cast to float tensor for training
        train_set = dset.MNIST(root=root, train=True, transform=trans, download=True)
        test_set = dset.MNIST(root=root, train=False, transform=trans, download=True)
        output_dim = np.array([28, 28])

    if psf:

        # -- first check if dataset exists
        dataset = simulate_propagated_dataset(
            psf=psf,
            sensor=sensor,
            output_dir=output_dir,
            down_psf=down_psf,
            down_out=down_out,
            output_dim=output_dim,
            scene2mask=scene2mask,
            mask2sensor=mask2sensor,
            object_height=object_height,
            device=device,
            crop_output=crop_output,
            grayscale=not rgb,
            single_psf=single_psf,
            n_files=n_files,
            crop_psf=crop_psf,
            return_output_dir=True,
            noise_type=noise_type,
            snr=snr,
        )
        if os.path.isdir(dataset):
            print(f"\nDataset already exists: {dataset}")
        else:
            # -- create simulated dataset
            dataset = simulate_propagated_dataset(
                psf=psf,
                sensor=sensor,
                output_dir=output_dir,
                down_psf=down_psf,
                down_out=down_out,
                output_dim=output_dim,
                scene2mask=scene2mask,
                mask2sensor=mask2sensor,
                object_height=object_height,
                device=device,
                crop_output=crop_output,
                grayscale=not rgb,
                single_psf=single_psf,
                n_files=n_files,
                crop_psf=crop_psf,
                noise_type=noise_type,
                snr=snr,
            )
        print()

    if dataset:

        # -- determine mean and standard deviation (of training set)
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
    if hidden:
        model = SingleHidden(input_shape=output_dim, hidden_dim=hidden, multi_gpu=multi_gpu)
    else:
        model = MultiClassLogistic(input_shape=output_dim, multi_gpu=multi_gpu)
    if use_cuda:
        # model = model.cuda()
        model = model.to(device)

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
    timestamp = datetime.now().strftime("%d%m%Y_%Hh%M")
    if dataset is None:
        model_output_dir = f"./MNIST_original_{n_epoch}epoch_seed{seed}_{model.name()}_{timestamp}"
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
    }
    metadata_fp = model_output_dir / "metadata.json"
    with open(metadata_fp, "w") as fp:
        json.dump(metadata, fp)

    test_loss_fp = model_output_dir / "test_loss.npy"
    test_acc_fp = model_output_dir / "test_acc.npy"

    print("Start training...")
    start_time = time.time()
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
                x, target = x.cuda(), target.cuda()

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
                x, target = x.cuda(), target.cuda()

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
