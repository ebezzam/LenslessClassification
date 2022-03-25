"""
Save a dataset convolved with PSF and scaled accordingly.

Convolve with lens PSF
```
python scripts/save_simulated_dataset.py --psf psfs/lens.png --crop_output
```

With lensless PSF (tape)
```
python scripts/save_simulated_dataset.py --psf psfs/tape.png
```

With lensless PSF (SLM)
```
python scripts/save_simulated_dataset.py --psf psfs/adafruit.png
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
@click.option("--down_out", type=float, help="Factor by which to downsample output.", default=128)
@click.option("--n_files", type=int, default=None)
def save_simulated_dataset(psf, down_out, n_files, crop_output, rgb, single_psf):
    if torch.cuda.is_available():
        print("CUDA available, using GPU.")
        device = "cuda"
    else:
        print("CUDA not available, using CPU.")
        device = "cpu"

    ## -- create output directory
    psf_bn = os.path.basename(psf).split(".")[0]
    OUTPUT_DIR = os.path.join("data", f"MNIST_{psf_bn}_down{int(down_out)}")
    if n_files:
        OUTPUT_DIR += f"_{n_files}files"
    if rgb:
        OUTPUT_DIR += "_rgb"

    print("\nSimulated dataset will be saved to :", OUTPUT_DIR)

    grayscale = not rgb
    scene2mask = 40e-2
    mask2sensor = 4e-3
    object_height = 5e-2
    downsample_psf = 2

    # RPi sensor dimension, TODO pass as param for different sensors
    pixel_size = np.array([1.55e-6, 1.55e-6])
    sensor_shape = np.array([3040, 4056])
    sensor_size = pixel_size * sensor_shape

    # load dataset
    if down_out:
        output_dim = tuple((sensor_shape * 1 / down_out).astype(int))
    else:
        output_dim = None
    print("OUTPUT DIMENSION ", output_dim)
    ds_train = MNISTPropagated(
        psf_fp=psf,
        downsample_psf=downsample_psf,
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
        downsample_psf=downsample_psf,
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
        if os.path.isfile(output_fp):
            train_labels.append(torch.load(label_fp))
        else:
            data = ds_train[i]

            # save as viewable images
            img_data = data[0].cpu().clone().numpy().squeeze()
            if len(img_data) == 3:
                # RGB
                img_data = img_data.transpose(1, 2, 0)
            im = Image.fromarray(img_data)
            im.save(output_fp)
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

            # save as viewable images
            img_data = data[0].cpu().clone().numpy().squeeze()
            if len(img_data) == 3:
                # RGB
                img_data = img_data.transpose(1, 2, 0)
            im = Image.fromarray(img_data)
            im.save(output_fp)
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
