"""

Train SLM as well as digital classifier.


"""

from collections import OrderedDict
from lenslessclass.models import SLMClassifier
import torch
import random
from pprint import pprint
import pathlib as plib
import os
import torch.nn as nn
from lenslessclass.datasets import Augmented
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as dset
import time
import click
from waveprop.devices import SLMOptions, SensorOptions, slm_dict, sensor_dict, SensorParam
import numpy as np
import json
from os.path import dirname, abspath, join
from datetime import datetime
from lenslessclass.datasets import (
    simulate_propagated_dataset,
    CELEBA_ATTR,
    CelebAAugmented,
    get_dataset_stats,
)
from lenslessclass.util import device_checks
from lenslessclass.vgg import cfg
from sklearn.model_selection import train_test_split
import pandas as pd
from torch.utils.data import Subset


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
    default=0.8,
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
@click.option("--lr", type=float, help="Learning rate.", default=None)
@click.option("--momentum", type=float, help="Momentum for SGD.", default=0.01)
@click.option("--n_epoch", type=int, help="Number of epochs to train.", default=50)
@click.option("--batch_size", type=int, help="Batch size.", default=32)
@click.option(
    "--print_epoch",
    type=int,
    help="How many batches to wait before printing epoch progress.",
    default=None,
)
@click.option(
    "--sensor_act",
    type=str,
    help="Activation at sensor. If not provided, none will applied.",
    default="relu",
)
@click.option(
    "--dropout",
    default=None,
    type=float,
    help="Percentage of dropout after first linear layer (if there's on that follows).",
)
@click.option(
    "--opti",
    type=click.Choice(["sgd", "adam"], case_sensitive=False),
    help="Optimizer.",
    default="adam",
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
    "--noise_type",
    default=None,
    type=click.Choice(["speckle", "gaussian", "s&p", "poisson"]),
    help="Noise type to add.",
)
@click.option(
    "--down_orig",
    type=float,
    default=1,
    help="Amount to downsample original number of input dimensions.",
)
@click.option("--snr", default=40, type=float, help="SNR to determine noise to add.")
@click.option("--down_out", type=float, help="Factor by which to downsample output.", default=None)
@click.option(
    "--down",
    default="resize",
    type=click.Choice(["resize", "max", "avg"], case_sensitive=False),
    help="Method for downsampling / reducing dimension.",
)
@click.option(
    "--down_psf", type=float, help="Factor by which to downsample convolution.", default=8
)
@click.option("--n_files", type=int, default=None)
@click.option("--device", type=str, help="Main device for training.")
@click.option("--device_mask", type=str, help="Main device for training.")
@click.option(
    "--single_gpu",
    is_flag=True,
    help="Whether to use single GPU is multiple available. Default will try using all.",
)
@click.option(
    "--hidden", type=int, default=None, help="If defined, add a hidden layer with this many units."
)
@click.option(
    "--hidden2",
    type=int,
    default=None,
    help="If defined, add a second hidden layer with this many units.",
)
@click.option(
    "--min_val",
    type=float,
    default=None,
    help="Minimum value from train set to make images non-negative. Default is to determine.",
)
@click.option(
    "--fix_slm",
    type=int,
    default=-1,
    help="Fix SLM make for this many epochs at the start.",
)
@click.option(
    "--shift",
    is_flag=True,
    help="Whether to random shift object in scene.",
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
    "--output_dim",
    default=None,
    nargs=2,
    type=int,
    help="Output dimension (height, width) at sensor. Use this instead of `down_out` if provided",
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
@click.option(
    "--dataset_seed",
    type=int,
    help="Random seed for splitting dataset. If None default to `seed`",
    default=None,
)
def train_hybrid(
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
    down,
    down_psf,
    down_orig,
    print_epoch,
    sensor_act,
    opti,
    dropout,
    seed,
    noise_type,
    snr,
    cont,
    object_height,
    output_dir,
    n_files,
    device,
    single_gpu,
    hidden,
    min_val,
    fix_slm,
    shift,
    hidden2,
    random_height,
    rotate,
    perspective,
    use_max_range,
    cnn,
    n_mask,
    cnn_lite,
    pool,
    kernel_size,
    task,
    multi_psf,
    rgb,
    vgg,
    aug_pad,
    device_mask,
    sched,
    output_dim,
    # celeba param
    attr,
    celeba_root,
    test_size,
    dataset_seed,
):
    if n_files == 0:
        n_files = None

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

    # task dependent
    task = task.upper()
    if task == "MNIST":
        target_dim = np.array([28, 28])
        dataset_object = dset.MNIST
        n_class = 10
    elif task == "CIFAR10":
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
        multi_gpu = metadata["model_param"]["multi_gpu"]
        dropout = metadata["model_param"]["dropout"]
        if "down" in metadata.keys():
            down = metadata["down"]
        if "down_orig" in metadata.keys():
            down_orig = metadata["down_orig"]
        if "hidden" in metadata["model_param"].keys():
            hidden = metadata["model_param"]["hidden"]
        if "hidden2" in metadata["model_param"].keys():
            hidden2 = metadata["model_param"]["hidden2"]
        if "min_val" in metadata.keys():
            min_val = metadata["min_val"]
        if "n_kern" in metadata.keys():
            cnn = metadata["n_kern"]
        if "n_slm_mask" in metadata.keys():
            n_mask = metadata["n_slm_mask"]

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if dataset_seed is None:
        dataset_seed = seed

    device, use_cuda, multi_gpu, device_ids = device_checks(device=device, single_gpu=single_gpu)
    if device_mask is None:
        device_mask_creation = device
    else:
        device_mask_creation = device_mask

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

        label_idx = ds.attr_names.index(attr)
        labels = ds.attr[:, label_idx][:n_files]

        train_indices, test_indices, _, _ = train_test_split(
            range(n_files),
            labels,
            train_size=train_size,
            test_size=test_size,
            stratify=labels,
            random_state=dataset_seed,
        )

        print(f"\ntrain set - {len(train_indices)}")
        df_attr = pd.DataFrame(labels[train_indices])
        print(df_attr.value_counts() / len(df_attr))

        print(f"\ntest set - {len(test_indices)}")
        df_attr = pd.DataFrame(labels[test_indices])
        print(df_attr.value_counts() / len(df_attr))

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

    # check if need to create dataset
    if random_height is not None:
        random_height = np.array(random_height) * 1e-2
        object_height = None

    if dataset is None:
        dataset = simulate_propagated_dataset(
            dataset=task,
            down_psf=down_psf,
            sensor=sensor,
            down_out=None,  # done during training
            scene2mask=scene2mask,
            mask2sensor=mask2sensor,
            object_height=object_height,
            device=device,
            crop_output=False,
            grayscale=not rgb,
            single_psf=not multi_psf,
            output_dir=output_dir,
            n_files=n_files,
            random_shift=shift,
            random_height=random_height,
            rotate=rotate,
            perspective=perspective,
            use_max_range=use_max_range,
        )

    def get_transform(mean, std, outdim, training=False, padding=0):
        """
        defaulted for CIFAR

        for VGG, do resizing to 32x32 inside model after sensor measurement
        """
        trans_list = []
        if task == "CIFAR10":
            if training:
                # augmentations like here: https://github.com/kuangliu/pytorch-cifar/blob/master/main.py#L31
                trans_list += [transforms.RandomHorizontalFlip()]
        if padding:
            trans_list += [transforms.RandomCrop(outdim, padding=padding)]
        trans_list += [transforms.ToTensor(), transforms.Normalize(mean, std)]
        return transforms.Compose(trans_list)

    # TODO -not needed for hybrid as we keep input non-negative for convolution with PSF
    # -- first determine mean and standard deviation (of training set)
    psf_dim = (sensor_param[SensorParam.SHAPE] / down_psf).astype(int)
    if mean is None or std is None:

        print("\nComputing stats...")

        if task == "CELEBA":
            trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0, 1)])
            all_data = CelebAAugmented(path=dataset, transform=trans)
            train_set = Subset(all_data, train_indices)
            mean, std = get_dataset_stats(train_set)

        else:

            train_set = Augmented(
                path=dataset, train=True, transform=get_transform(mean=0, std=1, outdim=psf_dim)
            )
            mean, std = train_set.get_stats(batch_size=batch_size)

        print("Dataset mean : ", mean)
        print("Dataset standard deviation : ", std)
        del train_set

    # -- normalize according to training set stats
    if task == "CELEBA":

        trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        all_data = CelebAAugmented(path=dataset, transform=trans)
        train_set = Subset(all_data, train_indices)
        test_set = Subset(all_data, test_indices)
        input_shape = list(train_set[0][0].shape)

    else:
        train_set = Augmented(
            path=dataset,
            train=True,
            transform=get_transform(
                mean=mean, std=std, outdim=psf_dim, training=True, padding=aug_pad
            ),
        )
        test_set = Augmented(
            path=dataset, train=False, transform=get_transform(mean=mean, std=std, outdim=psf_dim)
        )
        input_shape = train_set.get_image_shape()

    train_loader = torch.utils.data.DataLoader(
        dataset=train_set, batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set, batch_size=batch_size, shuffle=False
    )

    print(f"Network input dimension : {input_shape}")
    print(f"number training examples: {len(train_set)}")
    print(f"number test examples: {len(test_set)}")
    print("==>>> total training batch number: {}".format(len(train_loader)))
    print("==>>> total testing batch number: {}".format(len(test_loader)))

    # define model
    model = SLMClassifier(
        input_shape=input_shape,
        slm_config=slm_dict[slm],
        sensor_config=sensor_param,
        crop_fact=crop_fact,
        device=device,
        deadspace=not simple,
        scene2mask=scene2mask,
        mask2sensor=mask2sensor,
        device_mask_creation=device_mask_creation,
        sensor_activation=sensor_act_fn,
        multi_gpu=device_ids,
        dropout=dropout,
        noise_type=noise_type,
        snr=snr,
        target_dim=target_dim,
        down=down,
        n_class=n_class,
        hidden=hidden,
        hidden2=hidden2,
        output_dim=output_dim,
        n_kern=cnn,
        n_slm_mask=n_mask,
        pool=pool,
        kernel_size=kernel_size,
        cnn_lite=cnn_lite,
        grayscale=not rgb,
        vgg=vgg,
    )

    if use_cuda:
        model = model.to(device)

    if cont:
        state_dict_fp = str(cont / "state_dict.pth")
        state_dict = torch.load(state_dict_fp)

        # --- fix after parameter name change
        params = []
        for k, v in state_dict.items():
            if "conv_bn2" in k:
                params.append((k.replace("conv_bn2", "bn"), v))
            else:
                params.append((k, v))
        state_dict = OrderedDict(params)

        model.load_state_dict(state_dict)
        model.compute_intensity_psf()

    if fix_slm > 0:
        model.slm_vals.requires_grad = False
        model._psf = model._psf.detach()

    # set optimizer
    if opti == "sgd":
        if lr is None:
            lr = 0.01
        optimizer = optim.SGD(
            # model.parameters(),
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr,
            momentum=momentum,
        )
    elif opti == "adam":
        if lr is None:
            lr = 0.001
        # same default params
        optimizer = optim.Adam(
            # model.parameters(),
            filter(lambda p: p.requires_grad, model.parameters()),
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

    if min_val is None:
        min_val = 0
        for (x, target) in train_loader:
            if x.min() < min_val:
                min_val = x.min()
        min_val = float(min_val.item())
        print("Minimum value : ", min_val)

    ## save best model param
    timestamp = datetime.now().strftime("%d%m%Y_%Hh%M")

    if task == "CELEBA":
        bn = f"{attr}_{os.path.basename(dataset)}"
    else:
        bn = f"{os.path.basename(dataset)}"
    model_output_dir = f"./{bn}_outdim{model.n_hidden}_{n_epoch}epoch_sched{sched}_batch{batch_size}_seed{seed}_{model.name()}"

    if noise_type:
        model_output_dir += f"_{noise_type}{snr}"
    if model.downsample is not None:
        model_output_dir += f"_DS{down}"
    model_output_dir += f"_{timestamp}"

    model_output_dir = plib.Path(model_output_dir)
    model_output_dir.mkdir(exist_ok=True)
    model_file = model_output_dir / "state_dict.pth"
    test_loss_fp = model_output_dir / "test_loss.npy"
    train_loss_fp = model_output_dir / "train_loss.npy"
    test_acc_fp = model_output_dir / "test_acc.npy"
    train_acc_fp = model_output_dir / "train_acc.npy"

    metadata = {
        "dataset": join(dirname(dirname(abspath(__file__))), dataset),
        "mean": mean.tolist() if rgb else mean,
        "std": std.tolist() if rgb else std,
        "slm": slm,
        "down_orig": down_orig,
        "seed": seed,
        "sensor": sensor,
        "sensor_activation": sensor_act,
        "model": model.name(),
        "model_param": {
            "input_shape": input_shape,
            "crop_fact": crop_fact,
            "device": device,
            "deadspace": not simple,
            "scene2mask": scene2mask,
            "mask2sensor": mask2sensor,
            "device_mask_creation": device_mask_creation,
            "down": down,
            "target_dim": target_dim.tolist(),
            "multi_gpu": multi_gpu,
            "dropout": dropout,
            "hidden": hidden,
            "hidden2": hidden2,
            "n_kern": cnn,
            "n_slm_mask": n_mask,
            "pool": pool,
            "cnn_lite": cnn_lite,
            "kernel_size": kernel_size,
            "grayscale": not rgb,
            "vgg": vgg,
            "n_class": n_class,
        },
        "aug_pad": aug_pad,
        "batch_size": batch_size,
        "noise_type": noise_type,
        "snr": None if noise_type is None else snr,
        "min_val": min_val,
        "random_shift": shift,
        "dataset_seed": dataset_seed,
    }

    metadata_fp = model_output_dir / "metadata.json"
    with open(metadata_fp, "w") as fp:
        json.dump(metadata, fp)

    print(f"Model saved to : {str(model_output_dir)}")

    print("\nStart training...")
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

        if epoch == fix_slm:
            print("Unfreezing SLM layer...")
            model.slm_vals.requires_grad = True

            # best to overwrite optimizer: https://stackoverflow.com/a/55766749
            # optimizer.param_groups.append({"params": extra_params})
            if opti == "sgd":
                optimizer = optim.SGD(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    lr=lr,
                    momentum=momentum,
                )
            elif opti == "adam":
                # same default params
                optimizer = optim.Adam(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    lr=lr,
                    betas=(0.9, 0.999),
                    eps=1e-08,
                    weight_decay=0,
                    amsgrad=False,
                )

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

                # non-negative
                x -= min_val

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

        proc_time = (time.time() - start_time) / 60.0
        running_loss = running_loss / len(test_loader)
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
    train_hybrid()
