"""
Convolve with a PSF. Use `--down_out` to set output size.

Convolve with lens PSF
```
python scripts/create_augmented_example.py --psf psfs/lens.png --crop_output
```

With lensless PSF (tape)
```

```

With lensless PSF (SLM)
```
python scripts/create_augmented_example.py --psf psfs/adafruit.png
```

TODO : plot with dimensions

"""

import time
import click
import numpy as np
import matplotlib.pyplot as plt
from lenslessclass.datasets import MNISTPropagated
from lensless.plot import plot_image
from lensless.io import load_psf


@click.command()
@click.option(
    "--psf",
    type=str,
    help="Path to PSF.",
)
@click.option("--idx", type=int, help="Index from original dataset to simulate", default=50)
@click.option(
    "--down_psf",
    type=float,
    help="Factor by which to downsample PSF for more efficient simulation.",
    default=2,
)
@click.option("--down_out", type=float, help="Factor by which to downsample output.", default=128)
@click.option(
    "--gamma",
    default=None,
    type=float,
    help="Gamma factor for plotting.",
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
    help="Crop output before downsampling (needed for lens PSF).",
)
@click.option(
    "--normalize_plot",
    is_flag=True,
    help="Whether to normalize simulation plot.",
)
def create_augmented_example(
    psf, idx, down_psf, down_out, gamma, single_psf, rgb, crop_output, normalize_plot
):
    assert psf is not None

    grayscale = not rgb
    scene2mask = 40e-2
    mask2sensor = 4e-3
    object_height = 5e-2
    device = "cuda"

    # RPi sensor dimension
    pixel_size = np.array([1.55e-6, 1.55e-6])
    sensor_shape = np.array([3040, 4056])
    sensor_size = pixel_size * sensor_shape

    # load dataset
    if down_out:
        output_dim = tuple((sensor_shape * 1 / down_out).astype(int))
    else:
        output_dim = None
    ds = MNISTPropagated(
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
    )
    n_files = len(ds)
    print("\nNumber of files :", n_files)

    _, ax = plt.subplots(ncols=3, nrows=1, figsize=(15, 10))

    # original image
    print("\n-- Original image")
    original_image = ds.data[idx]
    print("device", original_image.device)
    print("shape", original_image.shape)
    print("dtype", original_image.dtype)
    print("minimum : ", original_image.min().item())
    print("maximum : ", original_image.max().item())
    plot_image(original_image, gamma=gamma, normalize=False, ax=ax[0])
    ax[0].set_title("Original")

    # plot PSF
    print("\n-- PSF")
    _psf = load_psf(fp=psf, return_float=True, verbose=True, single_psf=single_psf)
    plot_image(_psf, gamma=3, normalize=True, ax=ax[1])
    ax[1].set_title("PSF")

    # get image
    n_trials = 10
    start_time = time.time()
    for _ in range(n_trials):
        res = ds[idx]
    proc_time = (time.time() - start_time) / n_trials
    print("\n-- Input image")
    input_image, label = res
    print("label", label)
    print("device", input_image.device)
    print("shape", input_image.shape)
    print("dtype", input_image.dtype)
    print("minimum : ", input_image.min().item())
    print("maximum : ", input_image.max().item())

    print("\ntime to simulate [s] :", proc_time)
    print("time for whole dataset [m] :", proc_time * n_files / 60)

    # plot augmented example
    input_image_cpu = np.transpose(input_image.cpu(), (1, 2, 0))
    plot_image(input_image_cpu, gamma=gamma, normalize=normalize_plot, ax=ax[2])
    ax[2].set_title("Simulated")

    plt.show()


if __name__ == "__main__":
    create_augmented_example()
