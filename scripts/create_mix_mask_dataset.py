import numpy as np
import pathlib as plib
import torch
import random
import torch.nn as nn
from lenslessclass.models import SLMMultiClassLogistic
import json
import time
import pandas as pd
from waveprop.devices import slm_dict, sensor_dict
import os
import cv2
from PIL import Image
from waveprop.devices import slm_dict, sensor_dict, SensorParam
import click


# if using learned PSF, set random_masks to None and specify this dict
ROOT_DIR = ""
model_paths = [
    ROOT_DIR
    / plib.Path(
        "celeba_no_psf_down8_height0.27_NORM_100000files_scene2mask0.55_outdim768_Male_25epoch_seed0_SLM_SingleHidden800_poisson40.0_22102022_16h24"
    ),
    ROOT_DIR
    / plib.Path(
        "celeba_no_psf_down8_height0.27_NORM_100000files_scene2mask0.55_outdim768_Male_25epoch_seed1_SLM_SingleHidden800_poisson40.0_22102022_20h35"
    ),
    ROOT_DIR
    / plib.Path(
        "celeba_no_psf_down8_height0.27_NORM_100000files_scene2mask0.55_outdim768_Male_25epoch_seed2_SLM_SingleHidden800_poisson40.0_22102022_23h35"
    ),
    ROOT_DIR
    / plib.Path(
        "celeba_no_psf_down8_height0.27_NORM_100000files_scene2mask0.55_outdim768_Male_25epoch_seed3_SLM_SingleHidden800_poisson40.0_23102022_02h33"
    ),
    ROOT_DIR
    / plib.Path(
        "celeba_no_psf_down8_height0.27_NORM_100000files_scene2mask0.55_outdim768_Male_25epoch_seed4_SLM_SingleHidden800_poisson40.0_23102022_05h36"
    ),
    ROOT_DIR
    / plib.Path(
        "celeba_no_psf_down8_height0.27_NORM_100000files_scene2mask0.55_outdim768_Male_25epoch_seed5_SLM_SingleHidden800_poisson40.0_23102022_08h36"
    ),
    ROOT_DIR
    / plib.Path(
        "celeba_no_psf_down8_height0.27_NORM_100000files_scene2mask0.55_outdim768_Male_25epoch_seed6_SLM_SingleHidden800_poisson40.0_23102022_12h11"
    ),
    ROOT_DIR
    / plib.Path(
        "celeba_no_psf_down8_height0.27_NORM_100000files_scene2mask0.55_outdim768_Male_25epoch_seed7_SLM_SingleHidden800_poisson40.0_23102022_15h33"
    ),
    ROOT_DIR
    / plib.Path(
        "celeba_no_psf_down8_height0.27_NORM_100000files_scene2mask0.55_outdim768_Male_25epoch_seed8_SLM_SingleHidden800_poisson40.0_23102022_19h11"
    ),
    ROOT_DIR
    / plib.Path(
        "celeba_no_psf_down8_height0.27_NORM_100000files_scene2mask0.55_outdim768_Male_25epoch_seed9_SLM_SingleHidden800_poisson40.0_23102022_23h59"
    ),
]


@click.command()
@click.option(
    "--n_mask",
    default=1,
    type=int,
    help="Number of random masks to alternate between.",
)
@click.option("--learned", is_flag=True, help="Use learned masks.")
def create_dataset(n_mask, learned):

    OFFSET = 100000  # don't overlap with files used for training / testing
    N_FILES = 100000
    print_progress = 1000
    rgb = False
    CELEBA_DIR = "/scratch"
    PSF_OUTPUT_DIR = "psfs"
    output_dim = [24, 32]
    # output_dim = [192, 256]
    # output_dim = [3, 4]

    non_lin = True
    non_lin_range = [3, 6]
    down_out = 8
    slm = "adafruit"
    crop_fact = 0.8
    deadspace = True
    mask2sensor = 0.004
    device_mask_creation = "cpu"
    bit_depth = 12

    # assume hacker knows ideal object height and distance
    object_height = 0.27
    scene2mask = 55e-2

    # other sim parameters, don't need to be known by hacker
    device_conv = "cpu"
    # device_conv = "cuda:1"
    sensor = "rpi_hq"
    noise_type = "poisson"
    snr = 40
    grayscale = True
    single_psf = False
    use_max_range = True

    ## HOUSEKEEPING
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        print("CUDA available, using GPU.")
        device = "cuda"
    else:
        device = "cpu"
        print("CUDA not available, using CPU.")

    ## PREPARE PSFS AS FILES
    psfs = []
    if not learned:

        sensor_param = sensor_dict[sensor]
        sensor_size = sensor_param[SensorParam.SHAPE]
        if down_out > 1:
            sensor_size = (sensor_size * 1 / down_out).astype(int)
        print(f"Simulated sensor size : {sensor_size}")

        # generate random masks
        for i in range(n_mask):

            # set seed
            torch.manual_seed(i)
            random.seed(i)
            np.random.seed(i)

            # create mask using hybrid model
            model = SLMMultiClassLogistic(
                input_shape=sensor_size,
                slm_config=slm_dict[slm],
                sensor_config=sensor_param,
                crop_fact=crop_fact,
                device=device,
                deadspace=deadspace,
                scene2mask=scene2mask,
                mask2sensor=mask2sensor,
                device_mask_creation=device_mask_creation,
                n_class=10,  # doesn't matter
                grayscale=False,
                requires_grad=False,
            )

            # set new values
            slm_vals = model.slm_vals
            if non_lin:
                slm_vals = slm_vals ** np.random.uniform(non_lin_range[0], non_lin_range[1])
            model.set_slm_vals(slm_vals)

            # get PSF
            psf_sim = model.get_psf(numpy=True)
            psf_sim = np.transpose(psf_sim, (1, 2, 0))
            psf_sim /= psf_sim.max()

            # -- viewable version (8 bit)
            psf_sim_8bit = (psf_sim * 255).astype(dtype=np.uint8)
            fp = os.path.join(
                PSF_OUTPUT_DIR,
                f"simulated_adafruit_scene2mask{scene2mask}_deadspace{deadspace}_down_out{down_out}_nonlin{non_lin}_seed{i}_8bit.png",
            )
            cv2.imwrite(fp, cv2.cvtColor(psf_sim_8bit, cv2.COLOR_RGB2BGR))

            # -- as on RPi (12 bit depth on 16 bit)
            psf_sim *= 2**bit_depth - 1
            psf_sim = psf_sim.astype(dtype=np.uint16)
            fp = os.path.join(
                PSF_OUTPUT_DIR,
                f"simulated_adafruit_scene2mask{scene2mask}_deadspace{deadspace}_down_out{down_out}_nonlin{non_lin}_seed{i}.png",
            )
            cv2.imwrite(fp, cv2.cvtColor(psf_sim, cv2.COLOR_RGB2BGR))
            print("Saved simulated PSF to : ", fp)

            psfs.append(fp)

    else:

        # use learned masks
        print("Loading learned PSFs...")

        assert len(model_paths) >= n_mask, "Not enough learned masks."

        for _path in model_paths[:n_mask]:
            print(_path)
            f = open(str(_path / "metadata.json"))
            metadata = json.load(f)

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
                    "n_class": 1,
                    "multi_gpu": [0, 1] if metadata["model_param"]["multi_gpu"] else False,
                    "return_measurement": True,
                }
            )

            mask2sensor = metadata["model_param"]["mask2sensor"]

            model = SLMMultiClassLogistic(**metadata["model_param"])

            # -- load from state dict
            state_dict_fp = str(_path / "state_dict.pth")
            model.load_state_dict(torch.load(state_dict_fp))

            # recompute PSF for loaded SLM values
            model.grayscale = False
            model.compute_intensity_psf()

            # save to file to use by propagater
            learned_slm = model._psf.cpu().detach().numpy().squeeze()
            learned_slm = np.transpose(learned_slm, (1, 2, 0))
            learned_slm /= learned_slm.max()

            # -- viewable version (8 bit)
            psf_sim_8bit = (learned_slm * 255).astype(dtype=np.uint8)
            fp = os.path.join(PSF_OUTPUT_DIR, f"{os.path.basename(_path)}_8bit.png")
            cv2.imwrite(fp, cv2.cvtColor(psf_sim_8bit, cv2.COLOR_RGB2BGR))

            # -- cast to uint12 as on sensor
            bit_depth = 12
            learned_slm *= 2**bit_depth - 1
            learned_slm = learned_slm.astype(dtype=np.uint16)

            fp = os.path.join(PSF_OUTPUT_DIR, f"{os.path.basename(_path)}.png")
            cv2.imwrite(fp, cv2.cvtColor(learned_slm, cv2.COLOR_RGB2BGR))
            print("Saved learned PSF to : ", fp)

            psfs.append(fp)

    assert len(psfs) > 0
    print(f"\n-- Mixing {len(psfs)} masks...")

    ## INITIALIZE OUTPUT FOLDER
    if not learned:
        output_dir = f"celeba_{len(psfs)}_random_mixed_mask_nonlin{non_lin}_out{np.prod(output_dim)}_offset{OFFSET}_nfiles{N_FILES}"
    else:
        output_dir = f"celeba_{len(psfs)}_learned_mixed_mask_out{np.prod(output_dim)}_offset{OFFSET}_nfiles{N_FILES}"
    output_dir = plib.Path(output_dir)
    if not os.path.isdir(output_dir):
        output_dir.mkdir(exist_ok=True)
        print("\nSimulated dataset will be saved to :", output_dir)
    else:
        print(f"\nDataset already exists: {output_dir}")

    args = {
        "psf_fp": psfs,
        "downsample_psf": 1,
        "output_dim": output_dim,
        "scene2mask": scene2mask,
        "mask2sensor": mask2sensor,
        "sensor": sensor,
        "object_height": object_height,
        "offset": OFFSET,
        "n_files": N_FILES,
        "device_conv": device_conv,
        "crop_psf": False,
        "grayscale": grayscale,
        "vflip": False,
        "split": "all",
        "single_psf": single_psf,
        "root": CELEBA_DIR,
        "noise_type": noise_type,
        "snr": snr,
        "use_max_range": use_max_range,
    }

    metadata_fp = output_dir / "metadata.json"
    with open(metadata_fp, "w") as fp:
        json.dump(args, fp)

    # -- data subdir
    train_output = output_dir / "all"
    if not os.path.isdir(train_output):
        train_output.mkdir(exist_ok=True)
    train_labels = []

    ## LOOP THROUGH DATASET AND RANDOMLY PICK MASK
    # -- pass the list of PSFs and one will be picked randomly during propagation
    print("\nLoading dataset...")
    ds_aug = CelebAPropagated(**args)

    print("\nSimulating dataset...")
    # -- slow to use data loader
    # data_loader = torch.utils.data.DataLoader(
    #     dataset=ds_aug, batch_size=200, shuffle=False, num_workers=0
    # )
    # for batch_idx, batch in enumerate(data_loader, start=0):
    #     x, target = batch

    n_complete = len(list(train_output.glob("*.png")))

    def _simulate(i):
        img, label = ds_aug[i]

        # -- save files
        output_fp = train_output / f"img{i}.png"
        label_fp = train_output / f"label{i}"

        if os.path.isfile(output_fp) and os.path.isfile(label_fp):
            train_labels.append(torch.load(label_fp))
        else:
            img_data = img.cpu().numpy().squeeze()

            if img_data.dtype == np.uint8:
                # save as viewable images
                if len(img_data.shape) == 3:
                    # RGB
                    img_data = img_data.transpose(1, 2, 0)
                im = Image.fromarray(img_data)
                im.save(output_fp)
            else:
                # save as float data
                np.save(output_fp, img_data)

            # save label
            if hasattr(label, "__len__"):
                label = label.numpy().tolist()
            torch.save(label, label_fp)
            train_labels.append(label)

        if i % print_progress == (print_progress - 1):
            proc_time = time.time() - start_time
            print(f"{i + 1 - OFFSET} / {N_FILES} examples, {proc_time / 60} minutes")

    start_time = time.time()

    for i in range(OFFSET, OFFSET + n_complete):
        label_fp = train_output / f"label{i}"
        train_labels.append(torch.load(label_fp))
    for i in range(OFFSET + n_complete, OFFSET + N_FILES):
        _simulate(i)

    with open(train_output / "labels.txt", "w") as f:
        for item in train_labels:
            f.write("%s\n" % item)

    proc_time = time.time() - start_time
    print(f"Total time : {proc_time / 60.} minutes")


if __name__ == "__main__":
    create_dataset()
