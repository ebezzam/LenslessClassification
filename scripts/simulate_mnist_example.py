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


MNIST_ROOT_DIR = "data"
OUTPUT_DIR = "saved_images/mnist_examples"
dataset_idx = 3
n_class = 10
return_measurement = True  # when applying `forward`` of model for SLM approach


psf_dict = {
    # "lens": {
    #     "psf": "psfs/lens.png",
    #     "crop_psf": 100,
    #     "mask2sensor": 0.00753,
    #     "down_psf": 1,
    # },
    # "ca": {
    #     "psf": "psfs/simulated_mls63_mask2sensor0p0005_17052022_18h00_12bit.png",
    #     "crop_psf": False,
    #     "mask2sensor": 0.5e-3,
    #     "down_psf": 8,
    # },
    # # "ca": {
    # #     "psf": "psfs/simulated_mls63_mask2sensor0p004_17052022_18h01_12bit.png",
    # #     "crop_psf": False,
    # #     "mask2sensor": 4e-3,
    # #     "down_psf": 8,
    # # },
    # "diffuser": {
    #     "psf": "psfs/tape.png",
    #     "crop_psf": False,
    #     "mask2sensor": 4e-3,
    #     "down_psf": 8,
    # },
    # "adafruit": {
    #     "psf": "psfs/adafruit.png",
    #     "crop_psf": False,
    #     "mask2sensor": 4e-3,
    #     "down_psf": 8,
    # },
    # "adafruit_sim": {
    #     "psf": "psfs/simulated_adafruit_deadspaceTrue_15052022_21h04.png",
    #     "crop_psf": False,
    #     "mask2sensor": 4e-3,
    #     "down_psf": 8,
    # },
    # "learned_slm_fcnn_12": {
    #     "model": "models/mnist/MNIST_no_psf_down8_outdim12_50epoch_seed0_SLMSingleHidden800_poisson40.0_DSresize_12052022_12h21",
    # },
    "learned_slm_cnn_12": {
        "model": "MNIST_no_psf_down8_height0.12_outdim12_50epoch_seed0_SLM_CNNLite_10_FCNN800_poisson40.0_DSresize_30082022_05h21",
    }
}

output_dim_vals = [(48, 64), (24, 32), (12, 16), (6, 8), (3, 4)]

noise_type = "poisson"
snr = 40
object_height = 0.12
grayscale = True
single_psf = False
scene2mask = 40e-2

device_conv = "cpu"
# device_conv = "cuda:1"
sensor = "rpi_hq"

use_max_range = True
gamma = 2.2

start_time = time.time()
for _psf in psf_dict:
    print()
    print(_psf)

    if "model" in psf_dict[_psf].keys():
        print("-- learned PSF...")

        # opening JSON file
        model_dir = plib.Path(psf_dict[_psf]["model"])
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
                "return_measurement": return_measurement,
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
        # print("SLM dimensions : ", model.slm_vals.shape)
        # print("Number of SLM pixels : ", np.prod(model.slm_vals.shape))
        # print("PSF shape : ", learned_slm.shape)

        # -- cast to uint as on sensor
        bit_depth = 12
        learned_slm /= learned_slm.max()
        learned_slm *= 2**bit_depth - 1
        learned_slm = learned_slm.astype(dtype=np.uint16)

        fp = os.path.join(OUTPUT_DIR, f"{_psf}_psf_16bit.png")
        cv2.imwrite(fp, cv2.cvtColor(learned_slm, cv2.COLOR_RGB2BGR))
        print("Saved learned PSF to : ", fp)

        # loop over output dimensions
        for output_dim in output_dim_vals:
            print(output_dim)

            ds_aug = MNISTPropagated(
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
                single_psf=single_psf,
                root=MNIST_ROOT_DIR,
                noise_type=noise_type,
                snr=snr,
                use_max_range=use_max_range,
            )

            img, label = ds_aug[dataset_idx]
            im = Image.fromarray(img[0].cpu().numpy())
            # print(img.shape)
            im.save(os.path.join(OUTPUT_DIR, f"{_psf}_{output_dim[0]}_{output_dim[1]}.png"))

        # save PSF (normalize first)
        psf_data = ds_aug.psf[0].cpu().numpy()
        psf_data /= psf_data.max()
        psf_data = gamma_correction(psf_data, gamma)
        psf_data *= 255
        psf_data = psf_data.astype(dtype=np.uint8)
        im = Image.fromarray(psf_data)
        im.save(os.path.join(OUTPUT_DIR, f"{_psf}_psf.png"))

    else:
        for output_dim in output_dim_vals:
            print(output_dim)

            ds_aug = MNISTPropagated(
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
                single_psf=single_psf,
                root=MNIST_ROOT_DIR,
                noise_type=noise_type,
                snr=snr,
                use_max_range=use_max_range,
            )

            img, label = ds_aug[dataset_idx]
            im = Image.fromarray(img[0].cpu().numpy())
            im.save(os.path.join(OUTPUT_DIR, f"{_psf}_{output_dim[0]}_{output_dim[1]}.png"))

        # save PSF (normalize first)
        psf_data = ds_aug.psf[0].cpu().numpy()
        psf_data /= psf_data.max()
        psf_data = gamma_correction(psf_data, gamma)
        psf_data *= 255
        psf_data = psf_data.astype(dtype=np.uint8)
        im = Image.fromarray(psf_data)
        im.save(os.path.join(OUTPUT_DIR, f"{_psf}_psf.png"))

    print(time.time() - start_time)
