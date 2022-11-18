from lenslessclass.datasets import MNISTPropagated
import numpy as np
from PIL import Image
import os
import time
from lensless.util import gamma_correction
import cv2
import pathlib as plib
import json
import torch
import torch.nn as nn
from waveprop.devices import slm_dict, sensor_dict
from lenslessclass.models import SLMMultiClassLogistic
from lenslessclass.util import lenless_recovery
import click
from lenslessclass.datasets import MNISTPropagated, CelebAPropagated, CIFAR10Propagated


# specify which cameras to simulate
psf_dict = {
    "lens": {
        "psf": "psfs/lens.png",
        "crop_psf": 100,
        "mask2sensor": 0.00753,
        "down_psf": 1,
        "single_psf": True,
    },
    "ca": {
        "psf": "psfs/simulated_mls63_mask2sensor0p0005_17052022_18h00_12bit.png",
        "crop_psf": False,
        "mask2sensor": 0.5e-3,
        "down_psf": 8,
        "single_psf": True,
    },
    "diffuser": {
        "psf": "psfs/tape.png",
        "crop_psf": False,
        "mask2sensor": 4e-3,
        "down_psf": 8,
        "single_psf": True,
    },
    "adafruit": {
        "psf": "psfs/adafruit.png",
        "crop_psf": False,
        "mask2sensor": 4e-3,
        "down_psf": 8,
        "single_psf": False,  # as we have color filter
    },
    "adafruit_sim": {
        "psf": "psfs/simulated_adafruit_deadspaceTrue_15052022_21h04.png",
        "crop_psf": False,
        "mask2sensor": 4e-3,
        "down_psf": 8,
        "single_psf": False,  # as we have color filter
    },
}


learned_mnist = {
    # - lr
    "learned_mask_lr_768": {
        "model": "MNIST_no_psf_down8_height0.12_NORM_outdim768_50epoch_schedNone_batch100_seed0_SLM_MultiClassLogistic_poisson40.0_DSresize_08112022_12h21",
        "output_dim_vals": [(24, 32)],
        "single_psf": False,
    },
    "learned_mask_lr_192": {
        "model": "MNIST_no_psf_down8_height0.12_outdim192_50epoch_batch100_seed0_SLM_MultiClassLogistic_poisson40.0_DSresize_19102022_06h23",
        "output_dim_vals": [(12, 16)],
        "single_psf": False,
    },
    "learned_mask_lr_48": {
        "model": "MNIST_no_psf_down8_height0.12_outdim48_50epoch_batch100_seed0_SLM_MultiClassLogistic_poisson40.0_DSresize_19102022_06h24",
        "output_dim_vals": [(6, 8)],
        "single_psf": False,
    },
    "learned_mask_lr_12": {
        "model": "MNIST_no_psf_down8_height0.12_outdim12_50epoch_batch100_seed0_SLM_MultiClassLogistic_poisson40.0_DSresize_18102022_20h40",
        "output_dim_vals": [(3, 4)],
        "single_psf": False,
    },
    # - fcnn
    "learned_mask_fcnn_768": {
        "model": "MNIST_no_psf_down8_height0.12_NORM_outdim768_50epoch_sched20_batch64_seed0_SLM_SingleHidden800_poisson40.0_DSresize_03112022_14h01",
        "output_dim_vals": [(24, 32)],
        "single_psf": False,
    },
    "learned_mask_fcnn_192": {
        "model": "MNIST_no_psf_down8_height0.12_NORM_outdim192_50epoch_sched20_batch64_seed0_SLM_SingleHidden800_poisson40.0_DSresize_03112022_18h09",
        "output_dim_vals": [(12, 16)],
        "single_psf": False,
    },
    "learned_mask_fcnn_48": {
        "model": "MNIST_no_psf_down8_height0.12_NORM_outdim48_50epoch_sched20_batch64_seed0_SLM_SingleHidden800_poisson40.0_DSresize_03112022_23h21",
        "output_dim_vals": [(6, 8)],
        "single_psf": False,
    },
    "learned_mask_fcnn_12": {
        "model": "MNIST_no_psf_down8_height0.12_NORM_outdim12_50epoch_sched20_batch64_seed0_SLM_SingleHidden800_poisson40.0_DSresize_04112022_04h12",
        "output_dim_vals": [(3, 4)],
        "single_psf": False,
    },
    # - perturb
    "learned_mask_shift": {
        "model": "MNIST_no_psf_down8_height0.12_RandomShift_outdim768_50epoch_sched20_batch64_seed0_SLM_SingleHidden800_poisson40.0_DSresize_04112022_11h18",
        "output_dim_vals": [(24, 32)],
        "single_psf": False,
    },
    "learned_mask_rescale": {
        "model": "MNIST_no_psf_down8_height0.02-0.2_outdim768_50epoch_sched20_batch64_seed0_SLM_SingleHidden800_poisson40.0_DSresize_04112022_16h29",
        "output_dim_vals": [(24, 32)],
        "single_psf": False,
    },
    "learned_mask_rotate": {
        "model": "MNIST_no_psf_down8_height0.12_RandomRotate90.0_outdim768_50epoch_sched20_batch64_seed0_SLM_SingleHidden800_poisson40.0_DSresize_04112022_21h25",
        "output_dim_vals": [(24, 32)],
        "single_psf": False,
    },
    "learned_mask_perspective": {
        "model": "MNIST_no_psf_down8_height0.12_RandomPerspective0.5_outdim768_50epoch_sched20_batch64_seed0_SLM_SingleHidden800_poisson40.0_DSresize_05112022_01h58",
        "output_dim_vals": [(24, 32)],
        "single_psf": False,
    },
}

learned_celeba = {
    # -- Gender
    "learned_mask_gender_768": {
        "model": "celeba_no_psf_down8_height0.27_NORM_100000files_scene2mask0.55_outdim768_Male_50epoch_seed0_SLM_SingleHidden800_poisson40.0_21102022_17h11",
        "output_dim_vals": [(24, 32)],
        "single_psf": False,
    },
    "learned_mask_gender_12": {
        "model": "celeba_no_psf_down8_height0.27_NORM_100000files_scene2mask0.55_outdim12_Male_50epoch_seed0_SLM_SingleHidden800_poisson40.0_21102022_17h22",
        "output_dim_vals": [(3, 4)],
        "single_psf": False,
    },
    # -- Smiling
    "learned_mask_smiling_768": {
        "model": "celeba_no_psf_down8_height0.27_NORM_100000files_scene2mask0.55_outdim768_Smiling_50epoch_seed0_SLM_SingleHidden800_poisson40.0_22102022_07h16",
        "output_dim_vals": [(24, 32)],
        "single_psf": False,
    },
    "learned_mask_smiling_12": {
        "model": "celeba_no_psf_down8_height0.27_NORM_100000files_scene2mask0.55_outdim12_Smiling_50epoch_seed0_SLM_SingleHidden800_poisson40.0_22102022_13h55",
        "output_dim_vals": [(3, 4)],
        "single_psf": False,
    },
    # - 10 seeds
    "learned_mask_gender_0": {
        "model": "celeba_no_psf_down8_height0.27_NORM_100000files_scene2mask0.55_outdim768_Male_25epoch_seed0_SLM_SingleHidden800_poisson40.0_22102022_16h24",
        "output_dim_vals": [(24, 32)],
        "single_psf": False,
    },
    "learned_mask_gender_1": {
        "model": "celeba_no_psf_down8_height0.27_NORM_100000files_scene2mask0.55_outdim768_Male_25epoch_seed1_SLM_SingleHidden800_poisson40.0_22102022_20h35",
        "output_dim_vals": [(24, 32)],
        "single_psf": False,
    },
    "learned_mask_gender_2": {
        "model": "celeba_no_psf_down8_height0.27_NORM_100000files_scene2mask0.55_outdim768_Male_25epoch_seed2_SLM_SingleHidden800_poisson40.0_22102022_23h35",
        "output_dim_vals": [(24, 32)],
        "single_psf": False,
    },
    "learned_mask_gender_3": {
        "model": "celeba_no_psf_down8_height0.27_NORM_100000files_scene2mask0.55_outdim768_Male_25epoch_seed3_SLM_SingleHidden800_poisson40.0_23102022_02h33",
        "output_dim_vals": [(24, 32)],
        "single_psf": False,
    },
    "learned_mask_gender_4": {
        "model": "celeba_no_psf_down8_height0.27_NORM_100000files_scene2mask0.55_outdim768_Male_25epoch_seed4_SLM_SingleHidden800_poisson40.0_23102022_05h36",
        "output_dim_vals": [(24, 32)],
        "single_psf": False,
    },
    "learned_mask_gender_5": {
        "model": "celeba_no_psf_down8_height0.27_NORM_100000files_scene2mask0.55_outdim768_Male_25epoch_seed5_SLM_SingleHidden800_poisson40.0_23102022_08h36",
        "output_dim_vals": [(24, 32)],
        "single_psf": False,
    },
    "learned_mask_gender_6": {
        "model": "celeba_no_psf_down8_height0.27_NORM_100000files_scene2mask0.55_outdim768_Male_25epoch_seed6_SLM_SingleHidden800_poisson40.0_23102022_12h11",
        "output_dim_vals": [(24, 32)],
        "single_psf": False,
    },
    "learned_mask_gender_7": {
        "model": "celeba_no_psf_down8_height0.27_NORM_100000files_scene2mask0.55_outdim768_Male_25epoch_seed7_SLM_SingleHidden800_poisson40.0_23102022_15h33",
        "output_dim_vals": [(24, 32)],
        "single_psf": False,
    },
    "learned_mask_gender_8": {
        "model": "celeba_no_psf_down8_height0.27_NORM_100000files_scene2mask0.55_outdim768_Male_25epoch_seed8_SLM_SingleHidden800_poisson40.0_23102022_19h11",
        "output_dim_vals": [(24, 32)],
        "single_psf": False,
    },
    "learned_mask_gender_9": {
        "model": "celeba_no_psf_down8_height0.27_NORM_100000files_scene2mask0.55_outdim768_Male_25epoch_seed9_SLM_SingleHidden800_poisson40.0_23102022_23h59",
        "output_dim_vals": [(24, 32)],
        "single_psf": False,
    },
}


learned_cifar10 = {
    "learned_2916": {
        "model": "CIFAR10_no_psf_down8_height0.25_NORM_outdim2916_50epoch_batch32_seed0_SLM_VGG11_poisson40.0_DSresize_24102022_11h47",
        "output_dim_vals": [(27, 36)],
        "single_psf": False,
    },
    "learned_663": {
        "model": "CIFAR10_no_psf_down8_height0.25_NORM_outdim663_50epoch_sched20_batch32_seed0_SLM_VGG11_poisson40.0_DSresize_31102022_09h13",
        "output_dim_vals": [(13, 17)],
        "single_psf": False,
    },
    "learned_144": {
        "model": "CIFAR10_no_psf_down8_height0.25_NORM_outdim144_50epoch_sched20_batch32_seed0_SLM_VGG11_poisson40.0_DSresize_31102022_16h22",
        "output_dim_vals": [(6, 8)],
        "single_psf": False,
    },
    "learned_36": {
        "model": "CIFAR10_no_psf_down8_height0.25_NORM_outdim36_50epoch_batch32_seed0_SLM_VGG11_poisson40.0_DSresize_23102022_17h56",
        "output_dim_vals": [(3, 4)],
        "single_psf": False,
    },
}


@click.command()
@click.option(
    "--task",
    type=click.Choice(["mnist", "cifar10", "celeba"], case_sensitive=False),
    default="mnist",
)
@click.option("--n_files", type=int, default=5)
@click.option(
    "--recover",
    type=int,
    default=0,
    help="Whether to recover image using convex optimization. Specify number of iterations.",
)
@click.option(
    "--cam",
    type=str,
    help="Select one camera from `psf_dict`.",
)
@click.option(
    "--random_shift",
    is_flag=True,
    help="Whether to apply random shifts.",
)
@click.option(
    "--random_height",
    default=None,
    nargs=2,
    type=float,
    help="Random height range in cm. `2 20` is used in experiments, namely 2-20 cm.",
)
@click.option(
    "--random_rotate",
    default=False,
    type=float,
    help="Random degrees to rotate: (-rotate, rotate). 90 is used in experiments",
)
@click.option(
    "--perspective",
    type=float,
    help="Whether to apply perspective distortions. 0.5 is used in experiments.",
)
@click.option(
    "--model_root",
    type=str,
    default="models",
    help="Where to load models from.",
)
def simulate_examples(
    task, n_files, recover, cam, random_shift, perspective, random_height, random_rotate, model_root
):
    if random_height is not None:
        random_height = np.array(random_height) * 1e-2
        object_height = None

    task = task.lower()
    if task == "mnist":
        root = "data"
        dataset_object = MNISTPropagated
        n_class = 10
        OUTPUT_DIR = "saved_images/mnist_examples"
        output_dim_vals = [(24, 32), (12, 16), (6, 8), (3, 4)]
        object_height = 0.12
        grayscale = True
        scene2mask = 40e-2

        psf_dict.update(learned_mnist)

    elif task == "celeba":
        root = "/scratch"
        dataset_object = CelebAPropagated
        n_class = 1
        OUTPUT_DIR = "saved_images/celeba_examples"
        output_dim_vals = [(24, 32), (12, 16), (6, 8), (3, 4)]
        object_height = 0.27
        grayscale = True
        scene2mask = 55e-2

        psf_dict.update(learned_celeba)

    elif task == "cifar10":
        root = "data"
        dataset_object = CIFAR10Propagated
        n_class = 10
        OUTPUT_DIR = "saved_images/cifar10_examples"
        output_dim_vals = [(27, 36), (13, 17), (6, 8), (3, 4)]
        object_height = 0.25
        grayscale = False
        scene2mask = 40e-2

        psf_dict.update(learned_cifar10)

        assert not recover, "Recover not supported for CIFAR10 due to RGB."

    else:
        raise ValueError(f"Unsupported task : {task}")

    noise_type = "poisson"
    snr = 40
    device_conv = "cpu"
    sensor = "rpi_hq"
    use_max_range = True
    gamma = 2.2  # for PSF plotting

    # recover parameters if set to True
    tv = False

    start_time = time.time()
    for _psf in psf_dict:

        if cam is not None and _psf != cam:
            continue

        print()
        print(_psf)

        if "model" in psf_dict[_psf].keys():
            print("-- learned PSF...")

            # opening JSON file
            model_dir = task / plib.Path(psf_dict[_psf]["model"])
            model_dir = model_root / model_dir
            f = open(str(model_dir / "metadata.json"))
            metadata = json.load(f)

            assert metadata["model_param"]["scene2mask"] == scene2mask

            # create model instance
            use_cuda = torch.cuda.is_available()
            if use_cuda:
                print("CUDA available, using GPU.")
                device = "cuda"
            else:
                device = "cpu"
                print("CUDA not available, using CPU.")
            sensor_act_fn = None
            sensor_act = metadata["sensor_activation"]
            if sensor_act is not None:
                if sensor_act == "relu":
                    sensor_act_fn = nn.ReLU()
                elif sensor_act == "leaky":
                    sensor_act_fn = nn.LeakyReLU(float=0.1)
                elif sensor_act == "tanh":
                    sensor_act_fn = nn.Tanh()
                else:
                    raise ValueError("Not supported activation.")

            metadata["model_param"].update(
                {
                    "slm_config": slm_dict[metadata["slm"]],
                    "sensor_config": sensor_dict[metadata["sensor"]],
                    "sensor_activation": sensor_act_fn,
                    "n_class": n_class,
                    "return_measurement": True,  # when applying `forward`` of model for SLM approach
                    "multi_gpu": [0, 1] if metadata["model_param"]["multi_gpu"] else False,
                }
            )
            model = SLMMultiClassLogistic(**metadata["model_param"])
            if use_cuda:
                model = model.to(device)

            # -- load from state dict
            state_dict_fp = str(model_dir / "state_dict.pth")
            model.load_state_dict(torch.load(state_dict_fp))

            # recompute PSF for loaded SLM values
            model.grayscale = False
            model.compute_intensity_psf()

            # save to file as rest of PSFs
            learned_slm = model._psf.cpu().detach().numpy().squeeze()
            learned_slm = np.transpose(learned_slm, (1, 2, 0))

            # -- cast to uint as on sensor
            bit_depth = 12
            learned_slm /= learned_slm.max()
            learned_slm *= 2**bit_depth - 1
            learned_slm = learned_slm.astype(dtype=np.uint16)

            fp = os.path.join(OUTPUT_DIR, f"{_psf}_psf_16bit.png")
            cv2.imwrite(fp, cv2.cvtColor(learned_slm, cv2.COLOR_RGB2BGR))
            print("Saved learned PSF to : ", fp)

            # loop over output dimensions
            for output_dim in psf_dict[_psf]["output_dim_vals"]:

                ds_aug = dataset_object(
                    psf_fp=fp,
                    sensor=sensor,
                    downsample_psf=1,
                    output_dim=output_dim,
                    scene2mask=scene2mask,
                    mask2sensor=metadata["model_param"]["mask2sensor"],
                    object_height=object_height,
                    device=device_conv,
                    crop_psf=False,
                    grayscale=grayscale,
                    vflip=False,
                    train=False,
                    single_psf=psf_dict[_psf]["single_psf"],
                    root=root,
                    noise_type=noise_type,
                    snr=snr,
                    use_max_range=use_max_range,
                    random_shift=random_shift,
                    random_height=random_height,
                    rotate=random_rotate,
                    perspective=perspective,
                    # celeba specific
                    split="all",
                )

                for dataset_idx in range(n_files):

                    img, _ = ds_aug[dataset_idx]

                    if not grayscale:
                        img = img.cpu().numpy().transpose(1, 2, 0)
                    else:
                        img = img[0].cpu().numpy()

                    im = Image.fromarray(img)

                    fp = os.path.join(
                        OUTPUT_DIR, f"{_psf}_{output_dim[0]}_{output_dim[1]}_idx{dataset_idx}"
                    )
                    if random_shift:
                        fp += "_shift"
                    if random_height is not None:
                        fp += "_rescale"
                    if random_rotate:
                        fp += "_rotate"
                    if perspective:
                        fp += "_perspective"
                    fp += ".png"
                    im.save(fp)
                    print("saved : ", fp)

            # save PSF (normalize first)
            if not grayscale and not psf_dict[_psf]["single_psf"]:
                psf_data = ds_aug.psf.cpu().numpy().transpose(1, 2, 0)
            else:
                psf_data = ds_aug.psf[0].cpu().numpy()
            psf_data /= psf_data.max()
            psf_data = gamma_correction(psf_data, gamma)
            psf_data *= 255
            psf_data = psf_data.astype(dtype=np.uint8)
            im = Image.fromarray(psf_data)
            fp = os.path.join(OUTPUT_DIR, f"{_psf}_psf.png")
            im.save(fp)
            print("saved PSF : ", fp)

        else:
            for output_dim in output_dim_vals:

                ds_aug = dataset_object(
                    psf_fp=psf_dict[_psf]["psf"],
                    sensor=sensor,
                    downsample_psf=psf_dict[_psf]["down_psf"],
                    output_dim=output_dim,
                    scene2mask=scene2mask,
                    mask2sensor=psf_dict[_psf]["mask2sensor"],
                    object_height=object_height,
                    device=device_conv,
                    crop_psf=psf_dict[_psf]["crop_psf"],
                    grayscale=grayscale,
                    vflip=False,
                    train=False,
                    single_psf=psf_dict[_psf]["single_psf"],
                    root=root,
                    noise_type=noise_type,
                    snr=snr,
                    use_max_range=use_max_range,
                    random_shift=random_shift,
                    random_height=random_height,
                    rotate=random_rotate,
                    perspective=perspective,
                    # celeba specific
                    split="all",
                )

                for dataset_idx in range(n_files):
                    img, _ = ds_aug[dataset_idx]

                    if not grayscale:
                        img = img.cpu().numpy().transpose(1, 2, 0)
                    else:
                        img = img[0].cpu().numpy()
                    im = Image.fromarray(img)

                    fp = os.path.join(
                        OUTPUT_DIR, f"{_psf}_{output_dim[0]}_{output_dim[1]}_idx{dataset_idx}"
                    )
                    if random_shift:
                        fp += "_shift"
                    if random_height is not None:
                        fp += "_rescale"
                    if random_rotate:
                        fp += "_rotate"
                    if perspective:
                        fp += "_perspective"
                    im.save(fp + ".png")
                    print("saved : ", fp + ".png")

                    if recover and "lens" not in _psf:
                        psf = ds_aug.psf[0].cpu().numpy()
                        psf /= psf.max()

                        img_8bit = img.copy()
                        img = img_8bit - img_8bit.min()
                        img = img / img.max()
                        img_est = lenless_recovery(
                            psf=psf, img=img, min_iter=recover, max_iter=recover, tv=tv
                        )

                        # save
                        im = Image.fromarray((img_est * 255).astype(dtype=np.uint8))
                        fp += "_recover"
                        im.save(f"{fp}.png")
                        print("saved recovered : ", fp + ".png")

            # save PSF (normalize first)
            if not grayscale and not psf_dict[_psf]["single_psf"]:
                psf_data = ds_aug.psf.cpu().numpy().transpose(1, 2, 0)
            else:
                psf_data = ds_aug.psf[0].cpu().numpy()
            psf_data /= psf_data.max()
            psf_data = gamma_correction(psf_data, gamma)
            psf_data *= 255
            psf_data = psf_data.astype(dtype=np.uint8)
            im = Image.fromarray(psf_data)
            im.save(os.path.join(OUTPUT_DIR, f"{_psf}_psf.png"))

    print("processing time [s] : ", time.time() - start_time)


if __name__ == "__main__":
    simulate_examples()
