from lenslessclass.datasets import CelebAPropagated, prep_psf
import numpy as np
from PIL import Image
import os
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import cv2
import pathlib as plib
import torch
import torch.nn as nn
from waveprop.devices import slm_dict, sensor_dict, SensorParam
from lenslessclass.models import SLMClassifier
from lensless.util import resize
import random
import glob
import click
from joblib import Parallel, delayed
from waveprop.util import zero_pad
from lenslessclass.util import lenless_recovery


@click.command()
@click.option(
    "--output_dim",
    nargs=2,
    type=int,
    help="Output dimension (height, width).",
)
@click.option(
    "--n_files",
    default=10,
    type=int,
    help="Output dimension (height, width).",
)
@click.option("--diff_mask", is_flag=True, help="Different mask values at recovery.")
@click.option("--diff_dist", is_flag=True, help="Different mask to sensor distance at recovery.")
@click.option("--no_mask_sim", is_flag=True, help="Use mask values directly as PSF.")
@click.option(
    "--mask_based",
    is_flag=True,
    help="Mask-based recover. Elementwise recovery. Otherwise PSF-based (default).",
)
@click.option(
    "--n_jobs",
    type=int,
    default=15,
    help="Number of CPU jobs for parallelizing simulation over distance.",
)
@click.option(
    "--scene2mask",
    type=float,
    default=55e-2,
    help="Distance between object and camera.",
)
@click.option(
    "--object_height",
    type=float,
    default=0.27,
    help="Distance between object and camera.",
)
@click.option(
    "--min_iter",
    type=int,
    default=500,
    help="Minimum iterations.",
)
@click.option(
    "--max_iter",
    type=int,
    default=500,
    help="Maximum iterations.",
)
@click.option(
    "--save_raw",
    is_flag=True,
    help="Save raw measurement.",
)
@click.option(
    "--tv",
    type=float,
    default=None,
    help="If provided, add TV norm prior (smooth gradients) with as lambda.",
)
@click.option("--output_dir", type=str, default="data", help="Path to save recovered dataset.")
def cvx_recovery(
    output_dim,
    n_files,
    diff_mask,
    diff_dist,
    no_mask_sim,
    mask_based,
    n_jobs,
    scene2mask,
    object_height,
    save_raw,
    tv,
    min_iter,
    max_iter,
    output_dir,
):
    assert output_dim is not None

    if min_iter > max_iter:
        min_iter = max_iter

    # mask2sensor = 4e-3, scene2mask = 55e-2, object_height = 0.27
    roi = [[95, 290], [175, 335]]
    # roi = [[0, -1], [0, -1]]   # whole image
    if scene2mask != 55e-2:
        raise ValueError("determine different ROI with the notebook!")

    if diff_dist:
        # mask to sensor distance
        # result: doesn't affect performance
        dist_range = (1e-3, 7e-3)
    else:
        dist_range = None

    CELEBA_ROOT_DIR = "/scratch"
    PSF_OUTPUT_DIR = "psfs"
    seed = 0

    # simulation parameters
    # output_dim = tuple(np.array([24, 32]) * 8)
    mask = "adafruit"
    device_mask_creation = "cpu"
    sensor = "rpi_hq"
    mask2sensor = 4e-3
    noise_type = "poisson"
    device_conv = "cpu"
    grayscale = True  # Pycsou solver doesn't support RGB

    # derived parameters
    sensor_param = sensor_dict[sensor]
    sensor_size = sensor_param[SensorParam.SHAPE]
    sensor_size = tuple(np.array([24, 32]) * 16)

    ## HOUSEKEEPING
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        print("CUDA available, using GPU.")
        device = "cuda"
    else:
        device = "cpu"
        print("CUDA not available, using CPU.")

    # create a random PSF
    model = SLMClassifier(
        input_shape=sensor_size,
        slm_config=slm_dict[mask],
        sensor_config=sensor_param,
        crop_fact=0.8,
        device=device,
        deadspace=True,
        scene2mask=scene2mask,
        mask2sensor=mask2sensor,
        device_mask_creation=device_mask_creation,
        n_class=10,  # doesn't matter
        grayscale=False,
        requires_grad=False,
    )

    # save psf
    psf_fp_8bbit = os.path.join(PSF_OUTPUT_DIR, f"test_8bit.png")
    model.save_psf(fp=psf_fp_8bbit, bit_depth=8)
    psf_fp = os.path.join(PSF_OUTPUT_DIR, f"test_12bit.png")
    model.save_psf(fp=psf_fp, bit_depth=12)
    psf, _ = prep_psf(psf_fp, grayscale=grayscale, torch_tensor=False)

    """ Low-dimension recovery  """
    output_dir_lowres = f"celeba_recovered_scene2mask{scene2mask}_height{object_height}_{output_dim[0]}x{output_dim[1]}"
    if diff_mask:
        output_dir_lowres += "_diff_slm"
    if diff_dist:
        output_dir_lowres += "_diff_dist"
    if no_mask_sim:
        output_dir_lowres += "_no_slm_sim"
    if mask_based:
        output_dir_lowres += "_mask_based"
    if tv:
        output_dir_lowres += f"_tv{tv}"
    output_dir_lowres = output_dir / plib.Path(output_dir_lowres)

    if os.path.isdir(output_dir_lowres):

        print(f"\nRecovered dataset already exists : {output_dir_lowres}")
        n_files_complete = len(glob.glob(os.path.join(output_dir_lowres, "*.png")))
        print(f"-- {n_files_complete} files complete")

    else:
        output_dir_lowres.mkdir(exist_ok=True)
        print("\nRecovered dataset will be saved to :", output_dir_lowres)
        n_files_complete = 0

    if n_files_complete < n_files:

        complete_files = glob.glob(os.path.join(output_dir_lowres, "*.png"))
        complete_files = [int(os.path.basename(fp).split(".")[0]) for fp in complete_files]
        s = set(complete_files)
        missing_files = [x for x in np.arange(n_files) if x not in s]

        ds_aug = CelebAPropagated(
            psf_fp=psf_fp,
            downsample_psf=1,
            output_dim=output_dim,
            scene2mask=scene2mask,
            mask2sensor=mask2sensor,
            sensor=sensor,
            object_height=object_height,
            device_conv=device_conv,
            grayscale=grayscale,
            root=CELEBA_ROOT_DIR,
            noise_type=noise_type,
        )

        mask = None
        if no_mask_sim:
            psf_shape = tuple(model._psf.shape[1:])
            psf = model.slm_vals.cpu().numpy()
            psf = zero_pad(psf)
            psf = resize(psf, shape=psf_shape, interpolation=cv2.INTER_NEAREST)
            psf /= psf.max()

            fp = os.path.join(PSF_OUTPUT_DIR, f"mask_8bit.png")
            psf_8bit = (psf * 255).astype(dtype=np.uint8)
            cv2.imwrite(fp, cv2.cvtColor(psf_8bit, cv2.COLOR_RGB2BGR))

        def recover(i, psf, model, grayscale, ds_aug, min_iter, mask, dist_range, tv, max_iter):
            if diff_mask:
                # set new mask values and recompute PSF
                model.set_slm_vals(torch.rand(model.slm_vals.shape))
                psf_fp = os.path.join(PSF_OUTPUT_DIR, f"diff_12bit.png")
                model.save_psf(fp=psf_fp, bit_depth=12)
                psf, _ = prep_psf(psf_fp, grayscale=grayscale, torch_tensor=False)

            if diff_dist:
                _mask2sensor = np.random.uniform(low=dist_range[0], high=dist_range[1])
                model.set_mask2sensor(_mask2sensor)
                psf_fp = os.path.join(PSF_OUTPUT_DIR, f"diff_12bit.png")
                model.save_psf(fp=psf_fp, bit_depth=12)
                psf, _ = prep_psf(psf_fp, grayscale=grayscale, torch_tensor=False)

            if mask_based:
                mask = psf
                psf = None

            # get and prepare measurement
            img, _ = ds_aug[i]
            img_8bit = img.cpu().numpy().squeeze()
            img = img_8bit - img_8bit.min()
            img = img / img.max()
            if save_raw:
                im = Image.fromarray(img_8bit)
                fp = f"{i}_raw.png"
                fp = os.path.join(output_dir_lowres, fp)
                im.save(fp)

            # solve inverse problem
            img_est = lenless_recovery(
                psf=psf, img=img, min_iter=min_iter, mask=mask, tv=tv, max_iter=max_iter
            )

            # save estimate image
            im = Image.fromarray((img_est * 255).astype(dtype=np.uint8))
            fp = f"{i}.png"
            fp = os.path.join(output_dir_lowres, fp)
            im.save(fp)

        Parallel(n_jobs=n_jobs)(
            delayed(recover)(
                i, psf, model, grayscale, ds_aug, min_iter, mask, dist_range, tv, max_iter
            )
            for i in missing_files
        )

    """ Compute metrics """
    # compare with image before convolution
    ds_aug = CelebAPropagated(
        output_dim=sensor_size,
        scene2mask=scene2mask,
        mask2sensor=mask2sensor,
        sensor=sensor,
        object_height=object_height,
        grayscale=True,
        root=CELEBA_ROOT_DIR,
    )

    psnr = []
    ssim = []
    for i in range(n_files):

        hi_res, _ = ds_aug[i]
        hi_res = np.transpose(hi_res.cpu(), (1, 2, 0)).squeeze().numpy()
        hi_res = hi_res / hi_res.max()
        hi_res = hi_res[roi[0][0] : roi[0][1], roi[1][0] : roi[1][1]]

        # low res
        lo_res = np.array(Image.open(os.path.join(output_dir_lowres, f"{i}.png")))
        lo_res = lo_res[roi[0][0] : roi[0][1], roi[1][0] : roi[1][1]]
        lo_res = lo_res / lo_res.max()

        psnr.append(peak_signal_noise_ratio(hi_res, lo_res))
        ssim.append(structural_similarity(hi_res, lo_res))

    # print metrics
    print(f"\npsnr : {np.mean(psnr)}")
    print(f"ssim : {np.mean(ssim)}")


if __name__ == "__main__":
    cvx_recovery()
