from lenslessclass.generator import SingleHidden
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
    "--root",
    type=str,
    default="./data",
    help="Parent directory of MNIST dataset.",
)
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

# parameters for creating dataset
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
@click.option("--hidden", type=int, default=300, help="Hidden layer with this many units.")
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
    "--use_max_range",
    is_flag=True,
    help="Normalize simulated data to maximum bit depth. Otherwise random but no clipping.",
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
@click.option(
    "--cnn",
    default=None,
    type=int,
    help="Use CNN model as classifier. Argument denotes number of kernels.",
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
def train_decoder(
    root,
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
    use_max_range,
    cnn,
    cnn_lite,
    pool,
    kernel_size,
):
    if n_files == 0:
        n_files = None

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

        # use original dataset
        print("\nNo dataset nor PSF provided, using original MNIST dataset!\n")

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
        if output_dim is None:
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
            "use_max_range": use_max_range,
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
        train_set = MNISTAugmented(path=dataset, train=True, transform=trans, return_original=root)
        test_set = MNISTAugmented(path=dataset, train=False, transform=trans, return_original=root)
        output_dim = train_set.output_dim

    train_loader = torch.utils.data.DataLoader(
        dataset=train_set, batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set, batch_size=batch_size, shuffle=False
    )

    print("==>>> total training batch number: {}".format(len(train_loader)))
    print("==>>> total testing batch number: {}".format(len(test_loader)))

    # load model
    model = SingleHidden(input_shape=output_dim, hidden_dim=hidden)
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
    if cont:
        test_loss = list(np.load(str(cont / "test_loss.npy")))
        train_loss = list(np.load(str(cont / "train_loss.npy")))
        best_test_loss = np.max(test_loss)
        best_test_loss_epoch = np.argmin(test_loss) + 1
    else:
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
