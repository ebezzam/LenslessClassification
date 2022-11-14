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
from lenslessclass.generator import SingleHidden, Conv, FC2PretrainedStyleGAN, Conv3
import lpips
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import gc


"""
TODO
- command line arg for hidden or conv model
- print number of parameters
"""


@click.command()
@click.option(
    "--dataset",
    type=str,
    help="Path to dataset.",
)
@click.option("--n_epoch", type=int, help="Number of epochs to train.", default=10)
@click.option("--seed", type=int, help="Random seed.", default=0)
@click.option("--batch_size", type=int, help="Batch size.", default=100)

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
    default="Male",
    type=click.Choice(CELEBA_ATTR, case_sensitive=True),
    help="Attribute to split dataset in a stratified fashion.",
)
@click.option(
    "--loss",
    type=click.Choice(["mse", "lpips", "l1", "ssim"], case_sensitive=True),
    default="mse",
    help="Loss for training generator.",
)
@click.option(
    "--model",
    type=click.Choice(["fc", "conv", "pre"], case_sensitive=True),
    default=None,
    help="Model for generator",
)
@click.option(
    "--pretrained_fp",
    type=str,
    default="/scratch/stylegan2/pretrained/ffhq.pkl",
    help="Path to pretrained Style GAN 2 checkpoint.",
)

# parameters for creating dataset
@click.option(
    "--psf",
    type=str,
    help="Path to PSF.",
)
@click.option(
    "--output_dir",
    type=str,
    default="/scratch/LenslessClassification_data/celeba",
    help="Path to save augmented dataset.",
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
@click.option(
    "--unfreeze_gan",
    is_flag=True,
    help="Whether to train GAN weights as well.",
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
@click.option(
    "--lr",
    type=float,
    default=0.01,
    help="Initial learning rate.",
)
@click.option(
    "--cont",
    type=str,
    help="Path to training to continue.",
)
@click.option(
    "--opti",
    type=click.Choice(["sgd", "adam"], case_sensitive=False),
    help="Optimizer.",
    default="sgd",
)
@click.option(
    "--sched",
    type=int,
    help="After how many steps to reduce learning rate.",
)
@click.option(
    "--aug_pad",
    type=int,
    help="If provided, pad with amount and apply horizontal flip to training data.",
)
@click.option(
    "--n_train_loops",
    type=int,
    default=1,
    help="Number of times to loop data. To be used with augmentation.",
)
@click.option("--offset", type=int, help="Offset of CelebA files to begin training", default=100000)
def train_decoder(
    root,
    attr,
    test_size,
    dataset,
    seed,
    n_epoch,
    batch_size,
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
    loss,
    model,
    pretrained_fp,
    unfreeze_gan,
    lr,
    cont,
    opti,
    offset,
    sched,
    aug_pad,
    n_train_loops,
):
    if n_files == 0:
        n_files = None

    if cont:
        cont = plib.Path(cont)
        print(f"\nCONTINUTING TRAINING FOR {n_epoch} EPOCHS")
        f = open(str(cont / "metadata.json"))
        metadata = json.load(f)
        pprint(metadata)

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    if crop_psf:
        down_psf = 1

    device, use_cuda, multi_gpu, device_ids = device_checks(device=device, single_gpu=single_gpu)

    ## LOAD DATASET

    if dataset is None:
        # create if doesn't exist

        # -- simulate
        if "lens" in psf:
            assert crop_psf is not None

        sensor_param = sensor_dict[sensor]
        sensor_size = sensor_param[SensorParam.SHAPE]

        if output_dim is None:
            if down_out:
                output_dim = tuple((sensor_size / down_out).astype(int))

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
            offset=offset,
        )
        n_files_dataset = n_files

    else:

        # load augmented dataset
        f = open(str(plib.Path(dataset) / "metadata.json"))
        metadata = json.load(f)

        print("\nDataset metadata")
        pprint(metadata)
        offset = metadata["offset"]
        n_files_dataset = metadata["n_files"]
        output_dim = metadata["output_dim"]

    # -- load original to have same split
    trans_list = [transforms.ToTensor(), transforms.Normalize((0.5,), (1,))]
    if not rgb:
        trans_list += [transforms.Grayscale(num_output_channels=1)]

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

    # if n_files is None:
    #     n_files = len(ds)
    #     train_size = 1 - test_size
    # else:
    #     print(f"Using {n_files}")
    #     test_size = int(n_files * test_size)
    #     train_size = n_files - test_size
    n_test = int(n_files_dataset * test_size)
    n_train = n_files_dataset - n_test
    # print(n_train, n_test)

    label_idx = ds.attr_names.index(attr)
    labels = ds.attr[offset : offset + n_files_dataset, label_idx]
    train_indices, test_indices, _, _ = train_test_split(
        range(offset, offset + n_files_dataset),
        labels,
        train_size=n_train,
        test_size=n_test,
        stratify=labels,
        random_state=seed,
    )
    train_indices = np.array(train_indices)
    test_indices = np.array(test_indices)

    # get same first first files, no matter n_files
    if n_files_dataset != n_files:
        print(f"Using {n_files}")
    n_test = int(n_files * test_size)
    n_train = n_files - n_test
    train_indices = train_indices[:n_train]
    test_indices = test_indices[:n_test]

    # print(n_train, n_test)
    print(train_indices[:5])
    print(test_indices[:5])

    # -- first determine mean and standard deviation (of training set)
    if mean is None and std is None:
        trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0, 1)])
        print("\nComputing stats...")

        all_data = CelebAAugmented(path=dataset, transform=trans, offset=offset)
        train_set = Subset(all_data, train_indices - offset)
        mean, std = get_dataset_stats(train_set)
        print("Dataset mean : ", mean)
        print("Dataset standard deviation : ", std)

        del all_data

    # -- normalize according to training set stats
    if aug_pad:

        # apply padding just to train
        trans_list = [transforms.RandomHorizontalFlip()]
        if aug_pad > 0:
            print(output_dim)
            print("padding of : ", aug_pad)
            trans_list += [transforms.RandomCrop(output_dim, padding=aug_pad)]
        trans_list += [transforms.ToTensor(), transforms.Normalize(mean, std)]
        all_data = CelebAAugmented(
            path=dataset,
            transform=trans,
            return_original=root,
            target_dim=target_dim.tolist(),
            offset=offset,
        )
        train_set = Subset(all_data, train_indices - offset)

        # just normalization for test set
        trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        all_data = CelebAAugmented(
            path=dataset,
            transform=trans,
            return_original=root,
            target_dim=target_dim.tolist(),
            offset=offset,
        )
        test_set = Subset(all_data, test_indices - offset)

    else:
        trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        all_data = CelebAAugmented(
            path=dataset,
            transform=trans,
            return_original=root,
            target_dim=target_dim.tolist(),
            offset=offset,
        )
        train_set = Subset(all_data, train_indices - offset)
        test_set = Subset(all_data, test_indices - offset)

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
    if model == "fc":
        model = SingleHidden(
            input_shape=output_dim, hidden_dim=hidden, n_output=np.prod(target_dim)
        )
    elif model == "conv":
        # model = Conv2(input_shape=output_dim, hidden_dim=hidden, n_output=np.prod(target_dim))
        model = Conv3(input_shape=output_dim, hidden_dim=hidden, n_output=np.prod(target_dim))
        # model = Conv(input_shape=output_dim, hidden_dim=hidden, n_output=np.prod(target_dim))
    elif model == "pre":
        model = FC2PretrainedStyleGAN(
            input_shape=output_dim,
            hidden=[800, 800],
            fp=pretrained_fp,
            output_dim=target_dim.tolist(),
            grayscale=not rgb,
            freeze_gan=not unfreeze_gan,
        )

    model_name = model.name()
    if multi_gpu:
        model = nn.DataParallel(model, device_ids=device_ids)
    if use_cuda:
        model = model.to(device)

    if cont:
        state_dict_fp = str(cont / "state_dict.pth")
        model.load_state_dict(torch.load(state_dict_fp))

    # training metric
    max_loss = False
    if loss == "mse":
        criterion = nn.MSELoss()
    elif loss == "lpips":
        assert rgb
        criterion = lpips.LPIPS(net="vgg")
        if use_cuda:
            # only for PIPS
            criterion.cuda()
    elif loss == "l1":
        criterion = nn.L1Loss()
    elif loss == "ssim":
        max_loss = True
        criterion = StructuralSimilarityIndexMeasure().to(device)
    else:
        raise ValueError(f"unsupported loss : {loss}")
    # could use L1 loss or binary cross entropy (images within (0, 1)): https://www.v7labs.com/blog/autoencoders-guide
    # https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html

    # testing metrics
    test_metrics_func = {
        # "lpips": LearnedPerceptualImagePatchSimilarity(net_type='vgg').to(device),  # only for RGB, normalized to [-1, 1]
        "psnr": PeakSignalNoiseRatio().to(device),
        "ssim": StructuralSimilarityIndexMeasure().to(device),
    }

    # set optimizer
    if opti == "sgd":
        optimizer = optim.SGD(
            model.parameters(), lr=lr, momentum=0.9, weight_decay=0, maximize=max_loss
        )
    elif opti == "adam":
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=0,
            amsgrad=False,
            maximize=max_loss,
        )
    else:
        raise ValueError

    if sched:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=sched, gamma=0.1, verbose=True
        )
    else:
        scheduler = None

    print("\nModel parameters:")
    for name, params in model.named_parameters():
        print(name, "\t", params.size(), "\t", params.requires_grad)
    print()
    print("Optimizer's state_dict:")
    for var_name in optimizer.state_dict():
        print(var_name, "\t", optimizer.state_dict()[var_name])

    ## save best model param
    timestamp = datetime.now().strftime("%d%m%Y_%Hh%M")
    # TODO : change based on loss
    model_output_dir = f"./{os.path.basename(dataset)}_{n_epoch}epoch_batch{batch_size}_sched{sched}_seed{seed}_{model_name}_{loss}_{n_files}trainfiles_{timestamp}"
    model_output_dir = plib.Path(model_output_dir)
    model_output_dir.mkdir(exist_ok=True)
    model_file = model_output_dir / "state_dict.pth"

    metadata = {
        "dataset": {
            "path": join(dirname(dirname(abspath(__file__))), dataset),
            "n_files": n_files,
            "mean": mean,
            "std": std,
            "attr": attr,
        },
        "seed": seed,
        "timestamp (DDMMYYYY_HhM)": timestamp,
        "model": model_name,
        "target_dim": target_dim.tolist(),
        "batch_size": batch_size,
        "hidden_dim": hidden,
        "noise_type": noise_type,
        "snr": None if noise_type is None else snr,
        "offset": offset,
    }
    np.save(model_output_dir / "train_indices", train_indices)
    np.save(model_output_dir / "test_indices", test_indices)
    metadata_fp = model_output_dir / "metadata.json"
    with open(metadata_fp, "w") as fp:
        json.dump(metadata, fp)

    test_loss_fp = model_output_dir / "test_loss.npy"
    train_loss_fp = model_output_dir / "train_loss.npy"
    test_metrics_fp = model_output_dir / "test_metrics.npy"

    print(f"Model saved to : {str(model_output_dir)}")

    print("Start training...")
    start_time = time.time()
    if cont:
        test_loss = list(np.load(str(cont / "test_loss.npy")))
        train_loss = list(np.load(str(cont / "train_loss.npy")))
        test_metrics = np.load(str(cont / "test_metrics.npy"), allow_pickle="TRUE").item()
        best_test_loss = np.min(test_loss)
        best_test_loss_epoch = np.argmax(test_loss) + 1
    else:
        test_loss = []
        train_loss = []
        best_test_loss = np.inf
        best_test_loss_epoch = 0
        test_metrics = dict()
        for k in test_metrics_func:
            test_metrics[k] = []

    for epoch in range(n_epoch):

        # clean up
        for k in test_metrics_func:
            test_metrics_func[k].reset()

        # training
        model.train()
        running_loss = 0.0

        for train_loop in range(n_train_loops):

            for i, (x, _, x_orig) in enumerate(train_loader):

                # get inputs
                if use_cuda:
                    x, x_orig = x.to(device), x_orig.to(device)
                # x_orig = x_orig.view(-1, np.prod(x_orig.size()[1:]))

                # zero parameters gradients
                optimizer.zero_grad(set_to_none=True)

                # forward, backward, optimize
                out = model(x)

                # loss, backward and optimize
                loss = criterion(out, x_orig)
                if len(loss.size()) != 0:
                    loss = loss.mean()
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.detach().cpu().item()
                if i % batch_size == (batch_size - 1):  # print every X mini-batches
                    proc_time = (time.time() - start_time) / 60.0

                    batch_num = i + 1 + train_loop * len(train_loader)
                    print(
                        f"[{epoch + 1}, {batch_num:5d} / {len(train_loader) * n_train_loops}, {proc_time:.2f} min] running loss: {running_loss / batch_num:.3f}"
                    )

        _loss = running_loss / (len(train_loader) * n_train_loops)
        train_loss.append(_loss)
        proc_time = (time.time() - start_time) / 60.0
        print(f"[{proc_time:.2f} min] training loss: {_loss:.3f}")

        # testing
        model.eval()
        running_loss = 0.0
        test_metrics_running_loss = dict()
        for k in test_metrics_func:
            test_metrics_running_loss[k] = 0.0
        with torch.no_grad():
            for i, (x, _, x_orig) in enumerate(test_loader):

                # get inputs
                if use_cuda:
                    x, x_orig = x.to(device), x_orig.to(device)
                # x_orig = x_orig.view(-1, np.prod(x_orig.size()[1:]))

                # forward, and compute loss
                out = model(x)
                loss = criterion(out, x_orig)
                if len(loss.size()) != 0:
                    loss = loss.mean()
                running_loss += loss.detach().cpu().item()

                # compute test metrics
                for k in test_metrics_func:
                    test_metrics_running_loss[k] += (
                        test_metrics_func[k](out, x_orig).detach().cpu().item()
                    )

            proc_time = (time.time() - start_time) / 60.0
            _loss = running_loss / len(test_loader)
            print(
                "==>>> epoch: {}, {:.2f} min, test loss: {:.6f}".format(epoch + 1, proc_time, _loss)
            )
            test_loss.append(_loss)
            for k in test_metrics_func:
                metr = test_metrics_running_loss[k] / len(test_loader)
                print(f"test {k} : {metr}")
                test_metrics[k].append(metr)

        if _loss < best_test_loss:
            # save model param
            best_test_loss = _loss
            best_test_loss_epoch = epoch + 1
            torch.save(model.state_dict(), str(model_file))

        # save losses
        with open(test_loss_fp, "wb") as f:
            np.save(f, np.array(test_loss))
        with open(train_loss_fp, "wb") as f:
            np.save(f, np.array(train_loss))
        np.save(test_metrics_fp, test_metrics)

        if scheduler is not None:
            scheduler.step()

        # clean up
        # torch.cuda.empty_cache()
        gc.collect()
        del loss, x, x_orig

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
