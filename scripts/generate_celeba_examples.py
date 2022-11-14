import numpy as np
import pathlib as plib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from lenslessclass.generator import SingleHidden, Conv3, Conv, FC2PretrainedStyleGAN
from lenslessclass.datasets import CelebAAugmented
import json
from pprint import pprint
from torch.utils.data import Subset
from lenslessclass.util import device_checks
from PIL import Image
import os


root = "/scratch"
single_gpu = True
device = "cuda:1"
IDX = np.arange(0, 1)  # image to save
output_dir = "saved_images/celeba_decoder"
MODEL_DIR = "models/celeba_decoders"


models = {
    "fixed": {
        100: plib.Path(
            "celeba_1_learned_mixed_mask_out768_offset100000_nfiles100000_50epoch_batch4_schedNone_seed0_Conv3_10000_l1_100trainfiles_27102022_14h22"
        ),
        1000: plib.Path(
            "celeba_1_learned_mixed_mask_out768_offset100000_nfiles100000_50epoch_batch16_schedNone_seed0_Conv3_10000_l1_1000trainfiles_27102022_19h54"
        ),
        10000: plib.Path(
            "celeba_1_learned_mixed_mask_out768_offset100000_nfiles100000_50epoch_batch32_schedNone_seed0_Conv3_10000_l1_10000trainfiles_27102022_13h46"
        ),
        100000: plib.Path(
            "celeba_1_learned_mixed_mask_out768_offset100000_nfiles100000_50epoch_batch32_schedNone_seed0_Conv3_10000_l1_100000trainfiles_26102022_09h41"
        ),
    },
    "10": {
        100: plib.Path(
            "celeba_10_learned_mixed_mask_out768_offset100000_nfiles100000_50epoch_batch4_schedNone_seed0_Conv3_10000_l1_100trainfiles_27102022_21h17"
        ),
        1000: plib.Path(
            "celeba_10_learned_mixed_mask_out768_offset100000_nfiles100000_50epoch_batch16_schedNone_seed0_Conv3_10000_l1_1000trainfiles_27102022_21h37"
        ),
        10000: plib.Path(
            "celeba_10_learned_mixed_mask_out768_offset100000_nfiles100000_50epoch_batch32_schedNone_seed0_Conv3_10000_l1_10000trainfiles_27102022_22h23"
        ),
        100000: plib.Path(
            "celeba_10_learned_mixed_mask_out768_offset100000_nfiles100000_50epoch_batch32_schedNone_seed0_Conv3_10000_l1_100000trainfiles_26102022_17h16"
        ),
    },
    "100": {
        100: plib.Path(
            "celeba_100_random_mixed_mask_nonlinTrue_out768_offset100000_nfiles100000_50epoch_batch4_schedNone_seed0_Conv3_10000_l1_100trainfiles_28102022_01h54"
        ),
        1000: plib.Path(
            "celeba_100_random_mixed_mask_nonlinTrue_out768_offset100000_nfiles100000_50epoch_batch16_schedNone_seed0_Conv3_10000_l1_1000trainfiles_28102022_02h09"
        ),
        10000: plib.Path(
            "celeba_100_random_mixed_mask_nonlinTrue_out768_offset100000_nfiles100000_50epoch_batch32_schedNone_seed0_Conv3_10000_l1_10000trainfiles_28102022_02h33"
        ),
        100000: plib.Path(
            "celeba_100_random_mixed_mask_nonlinTrue_out768_offset100000_nfiles100000_50epoch_batch32_schedNone_seed0_Conv3_10000_l1_100000trainfiles_27102022_19h46"
        ),
    },
}

for n_mask in models.keys():

    for n_attack in models[n_mask].keys():

        model_dir = MODEL_DIR / models[n_mask][n_attack]
        print("\n--------------")
        print(model_dir)
        print("--------------\n")

        # Opening metadata file
        f = open(str(model_dir / "metadata.json"))
        metadata = json.load(f)
        pprint(metadata)

        # load test set
        offset = metadata["offset"] if "offset" in metadata.keys() else 0
        test_indices = np.load(model_dir / "test_indices.npy")
        trans = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(metadata["dataset"]["mean"], metadata["dataset"]["std"]),
            ]
        )
        all_data = CelebAAugmented(
            path=metadata["dataset"]["path"],
            transform=trans,
            return_original=root,
            target_dim=metadata["target_dim"],
            offset=offset,
        )
        test_set = Subset(all_data, test_indices - offset)
        print("embedding shape", test_set[0][0].shape)
        print("image shape", test_set[0][2].shape)

        # load model
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            print("CUDA available, using GPU.")
        else:
            device = "cpu"
            print("CUDA not available, using CPU.")
        device, use_cuda, multi_gpu, device_ids = device_checks(
            device=device, single_gpu=single_gpu
        )

        """ detect which model from name"""
        if "SingleHidden" in metadata["model"]:
            model = SingleHidden(
                input_shape=np.prod(test_set[0][0].shape),
                hidden_dim=metadata["hidden_dim"],
                n_output=np.prod(test_set[0][2].shape),
            )
        elif "Conv3" in metadata["model"]:
            model = Conv3(
                input_shape=np.prod(test_set[0][0].shape),
                hidden_dim=metadata["hidden_dim"],
                n_output=np.prod(test_set[0][2].shape),
            )
        elif "Conv" in metadata["model"]:
            model = Conv(
                input_shape=np.prod(test_set[0][0].shape),
                hidden_dim=metadata["hidden_dim"],
                n_output=np.prod(test_set[0][2].shape),
            )
        elif metadata["model"] == "FC2PretrainedStyleGAN_3":
            model = FC2PretrainedStyleGAN(
                input_shape=list(test_set[0][0].shape[1:]),
                hidden=[800, 800],
                fp="/scratch/stylegan2/pretrained/ffhq.pkl",
                output_dim=list(test_set[0][2].shape[1:]),
                grayscale=True,
            )
        if use_cuda:
            model = model.to(device)
        # need to call DataParallel as was trained with this
        model = nn.DataParallel(model, device_ids=device_ids)

        # -- load from state dict
        state_dict_fp = str(model_dir / "state_dict.pth")
        model.load_state_dict(torch.load(state_dict_fp))

        model.eval()

        print("\nModel's state_dict:")
        for param_tensor in model.state_dict():
            print(param_tensor, "\t", model.state_dict()[param_tensor].size())

        def normalize_and_uint8(img):
            img -= img.min()
            img /= img.max()
            img *= 255
            return img.astype(np.uint8)

        # get images
        plt.figure(figsize=(10, 10))
        for n, _idx in enumerate(IDX):
            # generate"../celeba_adafruit_outdim768_height0.27_scene2mask0.55_poisson40.0_downpsf8.0_NORM_100000offset_10000files_50epoch_seed0_Conv3_10000_mse_27092022_09h51"
            ex = test_set[_idx]
            gen_out = model(ex[0].to(device))
            gen_out_img = gen_out.detach().cpu().numpy().squeeze()

            # raw
            raw = normalize_and_uint8(ex[0].detach().cpu().numpy().squeeze())
            im = Image.fromarray(raw)
            fp = os.path.join(output_dir, f"{n_mask}_raw_{_idx}.png")
            im.save(fp)

            # decoder
            decoded = normalize_and_uint8(gen_out_img)
            im = Image.fromarray(decoded)
            fp = os.path.join(output_dir, f"{n_mask}_{n_attack}_decoded_{_idx}.png")
            im.save(fp)

            # original
            original = normalize_and_uint8(ex[2].cpu().numpy().squeeze())
            im = Image.fromarray(original)
            fp = os.path.join(output_dir, f"original_{_idx}.png")
            im.save(fp)

print("Images saved to : ", output_dir)
