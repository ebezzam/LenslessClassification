"""
Save a dataset convolved with PSF and scaled accordingly.

TODO : do in data loader so can be done faster in batches

Convolve with lens PSF
```
python scripts/save_simulated_dataset.py --psf psfs/lens.png --crop_output
```

Convolve with lensless PSF (tape)
```
python scripts/save_simulated_dataset.py --psf psfs/tape.png
```

Convolve with lensless PSF (SLM)
```
python scripts/save_simulated_dataset.py --psf psfs/adafruit.png
```

Resized and scaled so that can be convolved with PSF during training
```
python scripts/save_simulated_dataset.py --down_psf 2 --output_dir  \
--n_files 100
```

"""

import time
import click
import numpy as np
import pathlib as plib
import torch
import os
from PIL import Image
from lenslessclass.datasets import MNISTPropagated


BATCH = 1000  # how often to print progress


@click.command()
@click.option(
    "--psf",
    type=str,
    help="Path to PSF.",
)
@click.option(
    "--crop_output",
    is_flag=True,
    help="Crop output before downsampling (needed for lens PSF).",
)
@click.option(
    "--rgb",
    is_flag=True,
    help="Whether to do RGB simulation.",
)
@click.option(
    "--single_psf",
    is_flag=True,
    help="Same PSF for all channels (sum) or unique PSF for RGB.",
)
@click.option(
    "--down_psf", type=float, help="Factor by which to downsample convolution.", default=2
)
@click.option("--cpu", is_flag=True, help="Use CPU even if GPU if available.")
@click.option("--down_out", type=float, help="Factor by which to downsample output.", default=128)
@click.option(
    "--output_dim",
    default=None,
    nargs=2,
    type=int,
    help="Output dimension (height, width). Use this instead of `down_out` if provided",
)
@click.option("--n_files", type=int, default=None)
@click.option("--output_dir", type=str, default="data", help="Path to save augmented dataset.")
@click.option("--object_height", type=float, help="Object height.", default=5e-2)
@click.option("--scene2mask", type=float, default=0.4, help="Scene to SLM/mask distance in meters.")
@click.option(
    "--mask2sensor", type=float, default=0.004, help="SLM/mask to sensor distance in meters."
)
def save_simulated_dataset(
    psf,
    down_psf,
    down_out,
    n_files,
    crop_output,
    rgb,
    single_psf,
    cpu,
    output_dir,
    object_height,
    scene2mask,
    mask2sensor,
    output_dim,
):
    use_cuda = torch.cuda.is_available()
    if cpu:
        device = "cpu"
        use_cuda = False
    else:
        if use_cuda:
            device = "cuda"
            print("CUDA available, using GPU.")
        else:
            device = "cpu"
            print("CUDA not available, using CPU.")

    ## -- create output directory
    if psf is not None:
        psf_bn = os.path.basename(psf).split(".")[0]
        OUTPUT_DIR = os.path.join(output_dir, f"MNIST_{psf_bn}_down{int(down_out)}")
    else:
        # prior to convolution with PSF
        down_out = None
        OUTPUT_DIR = os.path.join(output_dir, f"MNIST_no_psf_down{int(down_psf)}")

    OUTPUT_DIR += f"_height{object_height}"
    if n_files:
        OUTPUT_DIR += f"_{n_files}files"
    if rgb:
        OUTPUT_DIR += "_rgb"

    print("\nSimulated dataset will be saved to :", OUTPUT_DIR)

    grayscale = not rgb

    # RPi sensor dimension, TODO pass as param for different sensors
    pixel_size = np.array([1.55e-6, 1.55e-6])
    sensor_shape = np.array([3040, 4056])
    sensor_size = pixel_size * sensor_shape

    # load dataset
    if down_out:
        output_dim = tuple((sensor_shape * 1 / down_out).astype(int))
    else:
        output_dim = tuple((sensor_shape * 1 / down_psf).astype(int))
    print("OUTPUT DIMENSION ", output_dim)
    print("Number of hidden units :", np.prod(output_dim))

    ds_train = MNISTPropagated(
        psf_fp=psf,
        downsample_psf=down_psf,
        output_dim=output_dim,
        scene2mask=scene2mask,
        mask2sensor=mask2sensor,
        sensor_size=sensor_size,
        object_height=object_height,
        device=device,
        crop_output=crop_output,
        grayscale=grayscale,
        vflip=False,
        train=True,
        single_psf=single_psf,
    )
    ds_test = MNISTPropagated(
        psf_fp=psf,
        downsample_psf=down_psf,
        output_dim=output_dim,
        scene2mask=scene2mask,
        mask2sensor=mask2sensor,
        sensor_size=sensor_size,
        object_height=object_height,
        device=device,
        crop_output=crop_output,
        grayscale=grayscale,
        vflip=False,
        train=False,
        single_psf=single_psf,
    )

    ## loop over samples and save
    output_dir = plib.Path(OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True)

    # -- train set
    train_output = output_dir / "train"
    train_output.mkdir(exist_ok=True)
    train_labels = []
    start_time = time.time()
    for i in range(len(ds_train)):
        if i == n_files:
            break

        output_fp = train_output / f"img{i}.png"
        label_fp = train_output / f"label{i}"
        if os.path.isfile(output_fp) and os.path.isfile(label_fp):
            train_labels.append(torch.load(label_fp))
        else:
            data = ds_train[i]
            img_data = data[0].cpu().clone().numpy().squeeze()

            if img_data.dtype == np.uint8:
                # save as viewable images
                if len(img_data) == 3:
                    # RGB
                    img_data = img_data.transpose(1, 2, 0)
                im = Image.fromarray(img_data)
                im.save(output_fp)
            else:
                # save as float data
                np.save(output_fp, img_data)

            # save label
            torch.save(data[1], label_fp)
            train_labels.append(data[1])

        if i % BATCH == (BATCH - 1):
            proc_time = time.time() - start_time
            print(f"{i + 1} / {len(ds_train)} examples, {proc_time / 60} minutes")

    with open(train_output / "labels.txt", "w") as f:
        for item in train_labels:
            f.write("%s\n" % item)

    proc_time = time.time() - start_time
    print(f"Processing time [m] : {proc_time/ 60}")
    print("Finished training set\n")

    # -- test set
    test_output = output_dir / "test"
    test_output.mkdir(exist_ok=True)
    test_labels = []
    start_time = time.time()
    for i in range(len(ds_test)):
        if i == n_files:
            break

        output_fp = test_output / f"img{i}.png"
        label_fp = test_output / f"label{i}"
        if os.path.isfile(output_fp):
            test_labels.append(torch.load(label_fp))
        else:
            data = ds_test[i]
            img_data = data[0].cpu().clone().numpy().squeeze()

            if img_data.dtype == np.uint8:
                # save as viewable images
                if len(img_data) == 3:
                    # RGB
                    img_data = img_data.transpose(1, 2, 0)
                im = Image.fromarray(img_data)
                im.save(output_fp)

            else:
                # save as flaot data
                np.save(output_fp, img_data)

            # save label
            torch.save(data[1], label_fp)
            test_labels.append(data[1])

        if i % BATCH == (BATCH - 1):
            proc_time = time.time() - start_time
            print(f"{i + 1} / {len(ds_test)} examples, {proc_time / 60} minutes")

    with open(test_output / "labels.txt", "w") as f:
        for item in test_labels:
            f.write("%s\n" % item)

    proc_time = time.time() - start_time
    print(f"Processing time [m] : {proc_time / 60}")
    print("Finished test set")


if __name__ == "__main__":
    save_simulated_dataset()
