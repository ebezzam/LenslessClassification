"""
Compare simulated and measured PSFs.

```
python scripts/compare_simulated_measured.py
```

Simulation parameters can be set via command line.

Interesting enough, the speckle noise pattern looks similar to the measured PSF.

"""


import matplotlib.pyplot as plt
from lensless.io import load_psf
from lensless.plot import plot_image, pixel_histogram
from lensless.util import print_image_info
import torch
import numpy as np
from waveprop.devices import SLMOptions, SensorOptions, slm_dict, sensor_dict
from waveprop.slm import get_active_pixel_dim, get_slm_mask
from waveprop.spherical import spherical_prop
from waveprop.color import ColorSystem
from waveprop.rs import angular_spectrum
from skimage.util.noise import random_noise
import cv2
import click
import os
from datetime import datetime
from lensless.constants import RPI_HQ_CAMERA_BLACK_LEVEL


@click.command()
@click.option(
    "--measured", type=str, help="Path to measure mask pattern.", default="psfs/adafruit.png"
)
@click.option("--down", type=float, help="Factor by which to downsample PSF.", default=4)
@click.option("--gamma", type=float, help="Gamma factor for plotting.", default=5)
@click.option(
    "--save",
    is_flag=True,
    help="Whether to save simulated PSF.",
)
@click.option(
    "--save_noise",
    is_flag=True,
    help="Whether to save simulated noise PSF.",
)
@click.option("--slm", type=str, help="Which SLM to use.", default=SLMOptions.ADAFRUIT.value)
@click.option("--sensor", type=str, help="Which sensor to use.", default=SensorOptions.RPI_HQ.value)
@click.option(
    "--crop_fact",
    type=float,
    default=0.7,
    help="Fraction of sensor that is left uncropped, centered.",
)
@click.option("--scene2mask", type=float, default=0.4, help="Scene to SLM/mask distance in meters.")
@click.option(
    "--mask2sensor", type=float, default=0.004, help="SLM/mask to sensor distance in meters."
)
@click.option(
    "--no_plot",
    is_flag=True,
    help="No plot, e.g. if on server.",
)
@click.option(
    "--noise_type",
    default="speckle",
    type=click.Choice(["speckle", "gaussian", "s&p", "poisson"]),
    help="Gradient descent update method.",
)
@click.option("--noise_mean", default=0, type=float, help="Noise standard deviation.")
@click.option("--noise_std", default=0.01, type=float, help="Noise standard deviation.")
@click.option("--max_val", default=500, type=float, help="Maximum value of simulated PSF.")
def compare_simulated_measured(
    measured,
    down,
    gamma,
    save,
    no_plot,
    noise_type,
    scene2mask,
    mask2sensor,
    crop_fact,
    slm,
    sensor,
    noise_mean,
    noise_std,
    max_val,
    save_noise,
):

    if save or save_noise:
        timestamp = datetime.now().strftime("%d%m%Y_%Hh%M")
        if save:
            save = os.path.join("psfs", f"simulated_adafruit_down{int(down)}_{timestamp}.png")
        if save_noise:
            save_noise = os.path.join(
                "psfs", f"simulated_adafruit_noise_down{int(down)}_{timestamp}.png"
            )
    bit_depth = 12
    if noise_mean is None:
        noise_mean = RPI_HQ_CAMERA_BLACK_LEVEL / (2**bit_depth - 1)
    return_float = False

    # plotting param
    normalize = True
    if not no_plot:
        _, ax = plt.subplots(ncols=3, nrows=2, figsize=(15, 10))

    # -- measured, load without removing background and keeping as uint16
    psf_meas = load_psf(fp=measured, downsample=down, return_float=return_float, bg_pix=None)
    print("\nMeasured")
    print_image_info(psf_meas)

    large_pixels = psf_meas[psf_meas > max_val]
    n_large_pixels = len(large_pixels)
    n_total_pixels = np.prod(psf_meas.shape)
    print(f"percentage of large pixels : {n_large_pixels / n_total_pixels * 100}")

    if not no_plot:
        plot_image(psf_meas, gamma=gamma, normalize=normalize, ax=ax[0][0])
        pixel_histogram(psf_meas, nbits=bit_depth, ax=ax[1][0])

    # -- simulated
    slm_config = slm_dict[slm]
    sensor_config = sensor_dict[sensor]
    target_dim = psf_meas.shape[:2]
    deadspace = True
    dtype = torch.float32
    device = "cpu"

    if dtype == torch.float32:
        ctype = torch.complex64
    else:
        ctype = torch.complex128

    # TODO : one-shot method to just return PSF
    overlapping_mask_size, overlapping_mask_dim, n_active_slm_pixels = get_active_pixel_dim(
        sensor_config=sensor_config,
        sensor_crop=crop_fact,
        slm_config=slm_config,
    )
    if deadspace:
        slm_vals = torch.rand(*n_active_slm_pixels, dtype=dtype, device=device)
    else:
        slm_vals = torch.rand(*overlapping_mask_dim, dtype=dtype, device=device)

    color_system = ColorSystem.rgb()
    d1 = np.array(overlapping_mask_size) / target_dim

    # -- object to mask
    spherical_wavefront = spherical_prop(
        in_shape=target_dim,
        d1=d1,
        wv=color_system.wv,
        dz=scene2mask,
        return_psf=True,
        is_torch=True,
        device=device,
        dtype=dtype,
    )

    # -- multiply with mask
    mask = get_slm_mask(
        slm_config=slm_config,
        sensor_config=sensor_config,
        crop_fact=crop_fact,
        target_dim=target_dim,
        slm_vals=slm_vals,
        deadspace=deadspace,
        device=device,
        dtype=dtype,
    )
    u_in = mask * spherical_wavefront

    # -- mask to sensor
    psfs = torch.zeros(u_in.shape, dtype=ctype, device=device)
    for i in range(color_system.n_wavelength):
        psfs[i], _, _ = angular_spectrum(
            u_in=u_in[i], wv=color_system.wv[i], d1=d1, dz=mask2sensor, dtype=dtype, device=device
        )
    psf_clean = torch.square(torch.abs(psfs)).numpy().transpose(1, 2, 0)
    psf_clean /= np.linalg.norm(psf_clean.ravel())

    # add noise
    psf_sim = random_noise(
        psf_clean, mode=noise_type, clip=True, mean=noise_mean, var=noise_std**2
    )
    noise = psf_sim - psf_clean
    noise = np.clip(noise, a_min=0, a_max=1)

    # cast to uint as on sensor
    print("\nSimulated")
    psf_sim /= psf_sim.max()
    if max_val is not None:
        psf_sim *= max_val
    else:
        psf_sim *= 2**bit_depth - 1
    psf_sim = psf_sim.astype(dtype=np.uint16)
    if save:
        cv2.imwrite(save, cv2.cvtColor(psf_sim, cv2.COLOR_RGB2BGR))
        print("Saved simulated PSF to : ", save)
    print_image_info(psf_sim)

    print("\nNoise")
    noise /= noise.max()
    if max_val is not None:
        noise *= max_val
    else:
        noise *= 2**bit_depth - 1
    noise = noise.astype(dtype=np.uint16)
    if save_noise:
        cv2.imwrite(save, cv2.cvtColor(noise, cv2.COLOR_RGB2BGR))
        print("Saved simulated noise to : ", save_noise)
    print_image_info(noise)

    if not no_plot:
        plot_image(psf_sim, gamma=gamma, normalize=normalize, ax=ax[0][1])
        pixel_histogram(psf_sim, nbits=bit_depth, ax=ax[1][1])
        plot_image(noise, gamma=gamma, normalize=normalize, ax=ax[0][2])
        pixel_histogram(noise, nbits=bit_depth, ax=ax[1][2])
        plt.show()


if __name__ == "__main__":
    compare_simulated_measured()
