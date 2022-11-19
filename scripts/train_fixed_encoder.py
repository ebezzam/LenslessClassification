from lenslessclass.models import (
    BinaryLogistic,
    MultiClassLogistic,
    SingleHidden,
    FullyConnected,
    CNN,
    CNNLite,
)
from lenslessclass.vgg import VGG, cfg
from lenslessclass.datasets import (
    CelebAAugmented,
    CELEBA_ATTR,
    simulate_propagated_dataset,
    get_dataset_stats,
    Augmented,
)
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Subset
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
from sklearn.model_selection import train_test_split
import pandas as pd


@click.command()
@click.option(
    "--dataset",
    type=str,
    help="Path to dataset.",
)
@click.option(
    "--task",
    type=click.Choice(["mnist", "cifar10", "celeba"], case_sensitive=False),
    default="mnist",
)
@click.option("--lr", type=float, help="Learning rate.", default=0.001)
@click.option("--n_epoch", type=int, help="Number of epochs to train.", default=50)
@click.option("--seed", type=int, help="Random seed.", default=0)
@click.option("--batch_size", type=int, help="Batch size.", default=32)
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
    help="Output dimension (height, width) at sensor. Use this instead of `down_out` if provided",
)
@click.option("--scene2mask", type=float, default=0.4, help="Scene to SLM/mask distance in meters.")
@click.option(
    "--mask2sensor", type=float, default=0.004, help="SLM/mask to sensor distance in meters."
)
@click.option(
    "--down_psf", type=float, help="Factor by which to downsample convolution.", default=8
)
@click.option("--object_height", type=float, default=0.12, help="Object height in meters.")
@click.option(
    "--multi_psf",
    is_flag=True,
    help="Unique PSF for RGB. Otherwise sum channels (as done in DiffuserCam).",
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
    default="poisson",
    type=click.Choice(["speckle", "gaussian", "s&p", "poisson"]),
    help="Noise type to add.",
)
@click.option("--snr", default=40, type=float, help="SNR to determine noise to add.")
@click.option("--sensor", type=str, help="Which sensor to use.", default=SensorOptions.RPI_HQ.value)
@click.option("--n_files", type=int, default=None)
@click.option(
    "--hidden",
    type=int,
    default=None,
    multiple=True,
    help="If defined, add a hidden layer with this many units.",
)
@click.option("--device", type=str, help="Main device for training.")
@click.option(
    "--down_orig",
    default=1,
    type=float,
    help="Amount to downsample original input dimension (number of pixels) NOT each dimension.",
)
@click.option(
    "--single_gpu",
    is_flag=True,
    help="Whether to use single GPU is multiple available. Default will try using all.",
)
@click.option(
    "--mean",
    type=float,
    default=None,
    multiple=True,
    help="Mean of original dataset to normalize, if not provided it will be computed.",
)
@click.option(
    "--std",
    type=float,
    default=None,
    multiple=True,
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
    "--print_epoch",
    type=int,
    help="How many batches to wait before printing epoch progress.",
    default=None,
)
@click.option(
    "--kernel_size",
    default=3,
    type=int,
    help="Kernel size for CNN models.",
)
@click.option(
    "--vgg",
    default=None,
    type=click.Choice(cfg.keys()),
    help="Use VGG model.",
)
@click.option(
    "--aug_pad",
    default=0,
    type=int,
    help="Whether to pad and random crop. This argument says by how much to padd. Not recommneded for lensless",
)
@click.option(
    "--no_flip_hor",
    is_flag=True,
    help="Flip input horizontally during training.",
)
@click.option(
    "--no_bn",
    is_flag=True,
    help="No batch norm.",
)
@click.option(
    "--sched",
    type=int,
    help="After how many steps to reduce learning rate. If not provided, not learning rate schedule",
)
# celeba parameters
@click.option(
    "--celeba_root",
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
    type=click.Choice(CELEBA_ATTR + ["Hair"], case_sensitive=True),
    help="Attribute to predict.",
)
def train_fixed_encoder(
    dataset,
    task,
    seed,
    lr,
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
    multi_psf,
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
    vgg,
    aug_pad,
    no_flip_hor,
    no_bn,
    sched,
    print_epoch,
    # celeba param
    attr,
    celeba_root,
    test_size,
):
    if n_files == 0:
        n_files = None

    down_orig = float(down_orig)

    if len(mean) == 0:
        mean = None
        std = None
    if mean is not None:
        assert std is not None
        if rgb:
            assert len(mean) == 3
            assert len(std) == 3
            mean = np.array(mean)
            std = np.array(std)
        else:
            assert len(mean) == 1
            assert len(std) == 1
            mean = mean[0]
            std = std[0]

    if crop_psf:
        assert down_psf == 1

    # task dependent
    task = task.upper()
    if task == "MNIST":
        root = "./data"
        target_dim = np.array([28, 28])
        dataset_object = dset.MNIST
        n_class = 10
    elif task == "CIFAR10":
        root = "./data"
        target_dim = np.array([32, 32])
        dataset_object = dset.CIFAR10
        n_class = 10
    elif task == "CELEBA":
        target_dim = np.array([218, 178])
        dataset_object = dset.CelebA
        n_class = 1
        assert attr is not None
    else:
        raise ValueError(f"Unsupported task : {task}")
    orig_dim = target_dim.tolist()
    print(f"\nOriginal dimension : {orig_dim}")

    # parse arch params
    if len(hidden) == 0:
        hidden = None
    elif len(hidden) == 1 and hidden[0] == 0:
        hidden = None
    else:
        hidden = list(hidden)

    # continuing already started training
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
        down_orig = metadata["down_orig"]
        attr = metadata["attr"]

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    device, use_cuda, multi_gpu, device_ids = device_checks(device=device, single_gpu=single_gpu)

    if print_epoch is None:
        print_epoch = batch_size

    if down_orig > 1:

        n_input = np.prod(target_dim) / down_orig
        h = int(np.round(np.sqrt(n_input * target_dim[0] / target_dim[1])))
        w = int(np.round(target_dim[1] / target_dim[0] * h))

        target_dim = np.array([int(h), int(w)])
        print(f"New target dimension : {target_dim}")
        print(f"Flattened : {np.prod(target_dim)}")

    ## LOAD DATASET

    if task == "CELEBA":
        # does not have set train-test split,
        # so we split to have same distribution of `attr` in train and test
        trans_list = [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (1.0,)),
            transforms.Grayscale(num_output_channels=1),
        ]

        if down_orig > 1:
            trans_list.append(transforms.Resize(size=target_dim.tolist()))

        trans = transforms.Compose(trans_list)
        ds = dset.CelebA(
            root=celeba_root,
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

        if attr == "Hair":
            # multiclass
            # last column is for putting unknown, will overwrite in loop
            hair_labels = [
                "Bald",
                "Black_Hair",
                "Blond_Hair",
                "Brown_Hair",
                "Gray_Hair",
                "Wearing_Hat",
                "Young",
            ]
            label_idx = [ds.attr_names.index(_lab) for _lab in hair_labels]
            labels = ds.attr[:, label_idx][:n_files]
            for i in range(n_files):
                if labels[i][:-1].sum() == 0:
                    # no label -> unknown
                    labels[i, -1] = 1
                else:
                    labels[i, -1] = 0

                    if labels[i][:-1].sum() != 1:
                        # multilabel...
                        # some processing
                        if labels[i, 0]:
                            # if any bald
                            labels[i] = 0
                            labels[i, 0] = 1
                        elif labels[i, 4]:
                            # then gray
                            labels[i] = 0
                            labels[i, 4] = 1
                        elif labels[i, 2]:
                            # then blond
                            labels[i] = 0
                            labels[i, 2] = 1
                        elif labels[i, 3]:
                            # then brown
                            labels[i] = 0
                            labels[i, 3] = 1
                        elif labels[i, 5]:
                            # if wearing hat but know color, remove hat label
                            labels[i, 5] = 0
                        else:
                            raise ValueError(labels[i])

            hair_labels[-1] = "Unknown"
            print(hair_labels)
            labels = np.argmax(labels, axis=1)

            raise ValueError("Too unbalanced...")

        else:
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

        print(f"\ntrain set distribution")
        df_attr = pd.DataFrame(labels[train_indices])
        print(df_attr.value_counts() / len(df_attr))

        print(f"\ntest set distribution")
        df_attr = pd.DataFrame(labels[test_indices])
        print(df_attr.value_counts() / len(df_attr))

    if dataset is None and psf is None:

        # use original dataset
        print(f"\nNo dataset nor PSF provided, using original {task} dataset!\n")
        noise_type = None

        if task == "CELEBA":

            # loaded before
            train_set = Subset(ds, train_indices)
            test_set = Subset(ds, test_indices)

        else:

            if not os.path.exists(root):
                os.mkdir(root)
            if task == "MNIST":
                mean = np.array(1 * [0.5])
                std = np.array(1 * [1.0])
            elif task == "CIFAR10":
                mean = np.array(3 * [0.5])
                std = np.array(3 * [1.0])
            trans_list = [transforms.ToTensor(), transforms.Normalize(mean, std)]
            if down_orig:
                trans_list.append(transforms.Resize(size=target_dim.tolist()))
            trans = transforms.Compose(trans_list)
            train_set = dataset_object(root=root, train=True, transform=trans, download=True)
            test_set = dataset_object(root=root, train=False, transform=trans, download=True)

        output_dim = list(train_set[0][0].shape)

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
                output_dim = (sensor_size * 1 / down_out).astype(int).tolist()
            else:
                # determine sensor resolution so that it matches number
                # of target dimension pixels, and keep aspect ratio
                # of sensor
                n_input = np.prod(target_dim)
                h = int(np.sqrt(n_input * sensor_size[0] / sensor_size[1]))
                w = int(sensor_size[1] / sensor_size[0] * h)
                output_dim = (h, w)

        assert output_dim is not None

        print(f"Output dimension : {output_dim}")
        print(f"Downsampling factor : {sensor_size[1] / output_dim[1]}")

        # generate dataset from provided PSF
        args = {
            "dataset": task,
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
            "single_psf": not multi_psf,
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

        def get_transform(mean, std, outdim, training=False, padding=0):
            """
            defaulted for CIFAR
            """
            trans_list = []

            if task == "CIFAR10":
                if vgg:
                    # resize image to (32, 32) square, easier to work with for CNN arch
                    # https://stats.stackexchange.com/questions/240690/non-square-images-for-image-classification
                    min_dim = min(outdim)  # taking minimum as edge typically black due to cropping
                    trans_list += [transforms.CenterCrop(min_dim), transforms.Resize(32)]
                    outdim = 32
                if training:
                    # augmentations like here: https://github.com/kuangliu/pytorch-cifar/blob/master/main.py#L31
                    if not no_flip_hor:
                        trans_list += [transforms.RandomHorizontalFlip()]
            if padding:
                print(f"Add random input shifts : {padding}...")
                trans_list += [transforms.RandomCrop(outdim, padding=padding)]
            trans_list += [transforms.ToTensor(), transforms.Normalize(mean, std)]
            return transforms.Compose(trans_list)

        # -- determine mean and standard deviation (of training set)
        if mean is None or std is None:
            if task == "CELEBA":
                trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0, 1)])
                all_data = CelebAAugmented(path=dataset, transform=trans)
                train_set = Subset(all_data, train_indices)
                mean, std = get_dataset_stats(train_set)

            else:
                train_set = Augmented(
                    path=dataset,
                    train=True,
                    transform=get_transform(mean=0, std=1, outdim=output_dim),
                )
                mean, std = train_set.get_stats()
            print("Dataset mean : ", mean)
            print("Dataset standard deviation : ", std)

        # -- normalize according to training set stats
        if task == "CELEBA":
            trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
            all_data = CelebAAugmented(path=dataset, transform=trans)
            train_set = Subset(all_data, train_indices)
            test_set = Subset(all_data, test_indices)
            output_dim = list(train_set[0][0].shape)

        else:
            train_set = Augmented(
                path=dataset,
                train=True,
                transform=get_transform(
                    mean=mean, std=std, outdim=output_dim, training=True, padding=aug_pad
                ),
            )
            test_set = Augmented(
                path=dataset,
                train=False,
                transform=get_transform(mean=mean, std=std, outdim=output_dim),
            )
            output_dim = train_set.get_image_shape()

    train_loader = torch.utils.data.DataLoader(
        dataset=train_set, batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set, batch_size=batch_size, shuffle=False
    )

    print(f"Network input dimension : {output_dim}")
    print(f"number training examples: {len(train_set)}")
    print(f"number test examples: {len(test_set)}")
    print("==>>> total training batch number: {}".format(len(train_loader)))
    print("==>>> total testing batch number: {}".format(len(test_loader)))

    ## training
    if vgg:
        model = VGG(vgg, input_shape=output_dim)
    elif deepbig:
        model = FullyConnected(input_shape=output_dim, n_class=n_class, dropout=dropout)
    elif len(hidden) > 1:
        model = FullyConnected(
            input_shape=output_dim, hidden_dim=hidden, n_class=n_class, dropout=dropout
        )
    elif cnn_lite:
        assert len(hidden) == 1
        model = CNNLite(
            input_shape=output_dim,
            n_kern=cnn_lite,
            hidden=hidden[0],
            n_class=n_class,
            pool=pool,
            kernel_size=kernel_size,
            bn=not no_bn,
            dropout=dropout,
        )
    elif len(hidden) == 1:
        model = SingleHidden(
            input_shape=output_dim,
            hidden_dim=hidden[0],
            n_class=n_class,
            dropout=dropout,
            bn=not no_bn,
        )
    elif cnn:
        model = CNN(
            input_shape=output_dim,
            n_kern=cnn,
            n_class=n_class,
            pool=pool,
            kernel_size=kernel_size,
            bn=not no_bn,
            dropout=dropout,
        )
    else:
        if n_class > 1:
            model = MultiClassLogistic(input_shape=output_dim, n_class=n_class)
        else:
            model = BinaryLogistic(input_shape=output_dim)
    model_name = model.name()

    # - model to GPU
    if multi_gpu:
        model = nn.DataParallel(model, device_ids=device_ids)

    if use_cuda:
        model = model.to(device)

    if cont:
        state_dict_fp = str(cont / "state_dict.pth")
        model.load_state_dict(torch.load(state_dict_fp))

    # set optimizer
    # TODO : learning rate scheduler
    if opti == "sgd":
        # https://github.com/kuangliu/pytorch-cifar/blob/49b7aa97b0c12fe0d4054e670403a16b6b834ddd/main.py#L87
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    elif opti == "adam":
        # same default params
        optimizer = optim.Adam(
            model.parameters(),
            lr=lr,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=0,
            amsgrad=False,
        )
    else:
        raise ValueError("Invalid optimization approach.")

    if sched:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=sched, gamma=0.1, verbose=True
        )
    else:
        scheduler = None
        sched = None

    if n_class > 1:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCELoss()

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
        if task == "CELEBA":
            bn = f"{attr}_{task}"
        else:
            bn = f"{task}"

        model_output_dir = f"./{bn}_original_{int(np.prod(output_dim))}dim_{n_epoch}epoch_sched{sched}_batch{batch_size}_seed{seed}_{model_name}_{timestamp}"
    else:
        if task == "CELEBA":
            bn = f"{attr}_{os.path.basename(dataset)}"
        else:
            bn = f"{os.path.basename(dataset)}"

        model_output_dir = f"./{bn}_{n_epoch}epoch_sched{sched}_batch{batch_size}_seed{seed}_{model_name}_{timestamp}"

    model_output_dir = plib.Path(model_output_dir)
    model_output_dir.mkdir(exist_ok=True)
    model_file = model_output_dir / "state_dict.pth"

    metadata = {
        "dataset": join(dirname(dirname(abspath(__file__))), dataset)
        if dataset is not None
        else None,
        "seed": seed,
        "down_out": down_out,
        "down_orig": down_orig,
        "attr": attr,
        "mean": mean.tolist() if rgb else mean,
        "std": std.tolist() if rgb else std,
        "timestamp (DDMMYYYY_HhM)": timestamp,
        "model": model_name,
        "model_param": {"input_shape": output_dim, "multi_gpu": device_ids},
        "device_ids": device_ids,
        "batch_size": batch_size,
        "hidden_dim": hidden,
        "deepbig": deepbig,
        "pool": pool,
        "kernel_size": kernel_size,
        "noise_type": noise_type,
        "snr": None if noise_type is None else snr,
        "dropout": dropout,
        "single_gpu": single_gpu,
        "crop_psf": crop_psf,
        "output_dim": output_dim,
    }
    metadata_fp = model_output_dir / "metadata.json"
    with open(metadata_fp, "w") as fp:
        json.dump(metadata, fp)

    test_loss_fp = model_output_dir / "test_loss.npy"
    train_loss_fp = model_output_dir / "train_loss.npy"
    test_acc_fp = model_output_dir / "test_acc.npy"
    train_acc_fp = model_output_dir / "train_acc.npy"

    print(f"Model saved to : {str(model_output_dir)}")

    print("Start training...")
    start_time = time.time()
    if cont:
        test_loss = list(np.load(str(cont / "test_loss.npy")))
        train_loss = list(np.load(str(cont / "train_loss.npy")))
        test_accuracy = list(np.load(str(cont / "test_acc.npy")))
        train_accuracy = list(np.load(str(cont / "train_acc.npy")))
        best_test_acc = np.max(test_accuracy)
        best_test_acc_epoch = np.argmax(test_accuracy) + 1
    else:
        test_loss = []
        train_loss = []
        test_accuracy = []
        train_accuracy = []
        best_test_acc = 0
        best_test_acc_epoch = 0
    for epoch in range(n_epoch):

        # training
        model.train()
        running_loss = 0.0
        correct_cnt, total_cnt = 0, 0
        for i, (x, target) in enumerate(train_loader):
            # get inputs
            if use_cuda:
                x, target = x.to(device), target.to(device)

            if task == "CELEBA":
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

            # train accuracy
            if n_class > 1:
                _, pred_label = torch.max(out.data, 1)
                correct_cnt += (pred_label == target.data).sum()
            else:
                pred_label = out.round()
                correct_cnt += (pred_label == target).sum()
            total_cnt += x.data.size()[0]
            running_acc = (correct_cnt * 1.0 / total_cnt).item()

            # print statistics
            running_loss += loss.item()
            if (i + 1) % print_epoch == 0:  # print every `print_epoch` mini-batches
                proc_time = (time.time() - start_time) / 60.0
                print(
                    f"[{epoch + 1}, {i + 1:5d} / {len(train_loader)}, {proc_time:.2f} min] loss: {running_loss  / (i + 1):.3f}, acc: {running_acc:.3f}"
                )

        _loss = running_loss / len(train_loader)
        train_loss.append(_loss)
        train_acc = (correct_cnt * 1.0 / total_cnt).item()
        train_accuracy.append(train_acc)
        print(f"training loss : {_loss:.3f}")
        print(f"training accuracy : {train_acc:.3f}")

        # testing
        model.eval()
        correct_cnt, running_loss = 0, 0
        total_cnt = 0

        with torch.no_grad():
            for i, (x, target) in enumerate(test_loader):

                # get inputs
                if use_cuda:
                    x, target = x.to(device), target.to(device)

                if task == "CELEBA":
                    target = target[:, label_idx]
                    target = target.unsqueeze(1)
                    target = target.to(x)

                # forward, and compute loss
                out = model(x)
                loss = criterion(out, target)

                # accumulate loss and accuracy
                if n_class > 1:
                    _, pred_label = torch.max(out.data, 1)
                    correct_cnt += (pred_label == target.data).sum()
                else:
                    pred_label = out.round()
                    correct_cnt += (pred_label == target).sum()
                total_cnt += x.data.size()[0]

                running_loss += loss.item()

        _loss = running_loss / len(test_loader)
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
        with open(train_loss_fp, "wb") as f:
            np.save(f, np.array(train_loss))
        with open(test_acc_fp, "wb") as f:
            np.save(f, np.array(test_accuracy))
        with open(train_acc_fp, "wb") as f:
            np.save(f, np.array(train_accuracy))

        if scheduler is not None:
            scheduler.step()

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
