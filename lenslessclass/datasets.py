from lensless.io import load_psf
from lensless.util import resize, rgb2gray
import cv2
import torch
from PIL import Image
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import glob
from lenslessclass.util import RealFFTConvolve2D
from tqdm import tqdm
from waveprop.devices import sensor_dict, SensorParam
import time
import pathlib as plib
from waveprop.devices import sensor_dict, SensorParam, SensorOptions
from lensless.constants import RPI_HQ_CAMERA_BLACK_LEVEL
from skimage.util.noise import random_noise
import itertools
from scipy import ndimage
import json
from torch import nn
from lenslessclass.util import AddPoissonNoise


# https://github.com/pytorch/pytorch/issues/1494#issuecomment-305993854
from multiprocessing import set_start_method

try:
    set_start_method("spawn")
except RuntimeError:
    pass


# TODO : abstract parent class for DatasetPropagated
# TODO : take into account FOV


class MNISTAugmented(Dataset):
    """
    For loading an augmented dataset (simulated or measured).
    """

    def __init__(self, path, train=True, transform=None, dtype=torch.float32):
        self._path = path
        if train:
            self._subdir = os.path.join(path, "train")
        else:
            self._subdir = os.path.join(path, "test")
        self._n_files = len(glob.glob(os.path.join(self._subdir, "img*.png")))

        with open(os.path.join(self._subdir, "labels.txt")) as f:
            self._labels = [int(i) for i in f]

        assert self._n_files == len(self._labels)

        self.transform = transform
        self.dtype = dtype

        # get output shape
        img = Image.open(glob.glob(os.path.join(self._subdir, "img*"))[0])
        # horizontal and vertical size in pixels, whereas PyTorch expects (height, width)

        self.output_dim = np.array(img.size)[::-1]

    def get_stats(self, batch_size=100, num_workers=4):
        """
        Get mean and standard deviation.

        Example: https://kozodoi.me/python/deep%20learning/pytorch/tutorial/2021/03/08/image-mean-std.html

        """

        image_loader = DataLoader(
            self, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
        )

        # placeholders
        psum = torch.tensor([0.0])
        psum_sq = torch.tensor([0.0])

        # loop through images
        for inputs in tqdm(image_loader):
            psum += inputs[0].sum(axis=[0, 2, 3])
            psum_sq += (inputs[0] ** 2).sum(axis=[0, 2, 3])

        # pixel count
        count = self._n_files * np.prod(self.output_dim)

        # mean and std
        total_mean = psum / count
        total_var = (psum_sq / count) - (total_mean**2)
        total_std = torch.sqrt(total_var)

        return total_mean.item(), total_std.item()

    def __getitem__(self, index):

        img_path = os.path.join(self._subdir, f"img{index}.png")
        img = Image.open(img_path)

        if self.transform:
            img = self.transform(img)

        return img, self._labels[index]

    def __len__(self):
        return self._n_files


class MNISTPropagated(datasets.MNIST):
    def __init__(
        self,
        object_height,
        scene2mask,
        mask2sensor,
        sensor,
        psf_fp=None,
        single_psf=False,
        downsample_psf=None,
        output_dim=None,
        crop_psf=False,
        vflip=True,
        grayscale=True,
        device=None,
        root="./data",
        train=True,
        download=True,
        scale=(1, 1),
        max_val=255,
        dtype=np.float32,
        dtype_out=torch.uint8,  # simulate quantization of sensor
        noise_type=None,
        snr=40,
        down="resize",
        random_shift=False,
        random_height=None,
        rotate=False,
        perspective=False,
        **kwargs,
    ):
        """
        downsample_psf : float
            Downsample PSF to do smaller convolution.
        crop_psf : int
            To be used for lens PSF! How much to crop around peak of lens to extract the PSF.

        """
        self.dtype = dtype
        self.dtype_out = dtype_out
        self.max_val = max_val

        self.input_dim = np.array([28, 28])
        sensor_param = sensor_dict[sensor]
        sensor_size = sensor_param[SensorParam.SIZE]

        # -- load PSF
        if psf_fp is not None:
            psf = load_psf(fp=psf_fp, single_psf=single_psf, dtype=dtype)

            if crop_psf:
                # for compact support PSF like lens
                # -- keep full convolution
                self.conv_dim = np.array(psf.shape)

                # -- crop PSF around peak
                center = np.unravel_index(np.argmax(psf, axis=None), psf.shape)
                top = int(center[0] - crop_psf / 2)
                left = int(center[1] - crop_psf / 2)
                psf = psf[top : top + crop_psf, left : left + crop_psf]

            else:
                # for PSFs with large support, e.g. lensless
                if downsample_psf:
                    psf = resize(psf, 1 / downsample_psf, interpolation=cv2.INTER_CUBIC).astype(
                        dtype
                    )
                    if single_psf:
                        # cv2 drops the last dimension when it's 1..
                        psf = psf[:, :, np.newaxis]
                self.conv_dim = np.array(psf.shape)

            # reorder axis to [channels, width, height]
            if grayscale and not single_psf:
                psf = rgb2gray(psf).astype(dtype)
                psf = psf[np.newaxis, :, :]
                self.conv_dim[2] = 1
                # if psf.shape[0] == 3:
                #     psf = rgb2gray(psf).astype(dtype)
                # else:
                #     psf = psf.squeeze()
            else:
                psf = np.transpose(psf, (2, 0, 1))

            # cast as torch array
            psf = torch.tensor(psf, device=device)
        else:

            # No PSF, output dimensions is same as dimension (before) convolution
            psf = None
            assert output_dim is not None
            self.conv_dim = np.array(output_dim)

        self.psf = psf

        # processing steps
        # -- convert to tensor and flip image if need be
        transform_list = [np.array, transforms.ToTensor()]
        if vflip:
            transform_list.append(transforms.RandomVerticalFlip(p=1.0))

        # -- resize to convolution dimension and scale to desired height at object plane
        magnification = mask2sensor / scene2mask
        self.scene_dim = sensor_size / magnification

        if random_height is not None:
            # TODO combine with shifting which needs to know padding
            assert len(random_height) == 2
            assert random_height[0] < random_height[1]

            def random_scale(image):

                object_height = np.random.uniform(low=random_height[0], high=random_height[1])
                object_height_pix = int(
                    np.round(object_height / self.scene_dim[1] * self.conv_dim[1])
                )
                scaling = object_height_pix / self.input_dim[1]
                object_dim = (np.round(self.input_dim * scaling)).astype(int).tolist()
                image = transforms.Resize(size=object_dim)(image)

                # -- pad rest with zeros
                padding = self.conv_dim[:2] - object_dim
                left = padding[1] // 2
                right = padding[1] - left
                top = padding[0] // 2
                bottom = padding[0] - top
                image = transforms.Pad(padding=(left, top, right, bottom))(image)
                return image

            transform_list.append(random_scale)
        else:
            assert isinstance(object_height, float)
            object_height_pix = int(np.round(object_height / self.scene_dim[1] * self.conv_dim[1]))
            scaling = object_height_pix / self.input_dim[1]
            object_dim = (np.round(self.input_dim * scaling)).astype(int).tolist()
            transform_list.append(
                transforms.RandomResizedCrop(size=object_dim, ratio=(1, 1), scale=scale)
            )

            # -- pad rest with zeros
            padding = self.conv_dim[:2] - object_dim
            left = padding[1] // 2
            right = padding[1] - left
            top = padding[0] // 2
            bottom = padding[0] - top
            transform_list.append(transforms.Pad(padding=(left, top, right, bottom)))

        if rotate:
            # rotate around center
            transform_list.append(transforms.RandomRotation(degrees=rotate))

        if perspective:
            transform_list.append(transforms.RandomPerspective(distortion_scale=perspective, p=1.0))

        # -- random shift
        if random_shift:

            assert (
                random_height is None
            ), "Random height not supported with random shift, need padding info"

            def shift_within_sensor(image):
                hshift = int(np.random.uniform(low=-left, high=right))
                vshift = int(np.random.uniform(low=-bottom, high=top))
                return torch.roll(image, shifts=(vshift, hshift), dims=(1, 2))

            transform_list.append(shift_within_sensor)

        if not grayscale:
            transform_list.append(transforms.Lambda(lambda x: x.repeat(3, 1, 1)))

        transform = transforms.Compose(transform_list)

        # -- to do convolution on GPU (must faster)
        self._transform_post = None
        if self.psf is not None:
            self._transform_post = []

            conv_op = RealFFTConvolve2D(self.psf, img_shape=np.roll(self.conv_dim, shift=1))
            self._transform_post.append(conv_op)

            # -- resize to output dimension
            # more manageable to train on and perhaps don't need so many DOF
            if output_dim:
                if down == "resize":
                    self._transform_post.append(transforms.Resize(size=output_dim))
                elif down == "max" or down == "avg":
                    hidden = np.prod(output_dim)

                    # determine filter size, stride, and padding: https://androidkt.com/calculate-output-size-convolutional-pooling-layers-cnn/
                    k = int(np.ceil(np.sqrt(np.prod(self.conv_dim) / hidden)))
                    p = np.roots(
                        [4, 2 * np.sum(self.conv_dim), np.prod(self.conv_dim) - k**2 * hidden]
                    )
                    p = max(int(np.max(p)), 0) + 1
                    if down == "max":
                        pooler = nn.MaxPool2d(kernel_size=k, stride=k, padding=p)
                    else:
                        pooler = nn.AvgPool2d(kernel_size=k, stride=k, padding=p)
                    self._transform_post.append(pooler)
                else:
                    raise ValueError("Invalid downsampling approach.")

            if noise_type:
                if noise_type == "poisson":
                    transform_list.append(AddPoissonNoise(snr))
                else:
                    if sensor == SensorOptions.RPI_HQ.value:
                        bit_depth = 12
                        noise_mean = RPI_HQ_CAMERA_BLACK_LEVEL / (2**bit_depth - 1)
                    else:
                        noise_mean = 0
                    transform_list.append(AddNoise(snr, noise_type, noise_mean, dtype))

            self._transform_post = transforms.Compose(self._transform_post)

        self.device = device
        super().__init__(root=root, train=train, download=download, transform=transform)

    def __getitem__(self, index):

        res = super().__getitem__(index)
        if self.device:
            img = res[0].to(device=self.device)
        else:
            img = res[0]

        if self._transform_post:
            img = self._transform_post(img)

        # cast to uint8 as on sensor
        # if img.max() > 1:
        img /= img.max()
        # cast to uint8 as on sensor and use max range
        img *= self.max_val
        img = img.to(dtype=self.dtype_out)

        return img, res[1]


""""
CelebA: https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
"""

CELEBA_ATTR = [
    "5_o_Clock_Shadow",
    "Arched_Eyebrows",
    "Attractive",
    "Bags_Under_Eyes",
    "Bald",
    "Bangs",
    "Big_Lips",
    "Big_Nose",
    "Black_Hair",
    "Blond_Hair",
    "Blurry",
    "Brown_Hair",
    "Bushy_Eyebrows",
    "Chubby",
    "Double_Chin",
    "Eyeglasses",
    "Goatee",
    "Gray_Hair",
    "Heavy_Makeup",
    "High_Cheekbones",
    "Male",
    "Mouth_Slightly_Open",
    "Mustache",
    "Narrow_Eyes",
    "No_Beard",
    "Oval_Face",
    "Pale_Skin",
    "Pointy_Nose",
    "Receding_Hairline",
    "Rosy_Cheeks",
    "Sideburns",
    "Smiling",
    "Straight_Hair",
    "Wavy_Hair",
    "Wearing_Earrings",
    "Wearing_Hat",
    "Wearing_Lipstick",
    "Wearing_Necklace",
    "Wearing_Necktie",
    "Young",
]


class CelebAAugmented(Dataset):
    """
    For loading an augmented dataset (simulated or measured).
    """

    def __init__(self, path, transform=None, dtype=torch.float32):
        self._path = path
        self._subdir = os.path.join(path, "all")
        self._n_files = len(glob.glob(os.path.join(self._subdir, "img*.png")))

        self._labels = []
        with open(os.path.join(self._subdir, "labels.txt")) as f:
            for _labels in f:
                _labels = _labels.strip("]\n[").split(", ")
                self._labels.append([int(i) for i in _labels])
        self._labels = torch.tensor(self._labels)

        assert self._n_files == len(self._labels)

        self.transform = transform
        self.dtype = dtype

        # get output shape
        img = Image.open(glob.glob(os.path.join(self._subdir, "img*"))[0])
        # horizontal and vertical size in pixels, whereas PyTorch expects (height, width)
        self.output_dim = np.array(img.size)[::-1]

    def __getitem__(self, index):

        img_path = os.path.join(self._subdir, f"img{index}.png")
        img = Image.open(img_path)

        if self.transform:
            img = self.transform(img)

        return img, self._labels[index]

    def __len__(self):
        return self._n_files


class AddNoise:
    def __init__(self, snr, noise_type, noise_mean, dtype):
        self.snr = snr
        self.noise_type = noise_type
        self.noise_mean = noise_mean
        self.dtype = dtype

    def __call__(self, measurement):
        # sig_var = np.linalg.norm(measurement)

        # if measurement.max() > 1:
        # normalize to 1 as mean is also normalized like so
        measurement /= measurement.max()

        measurement_np = measurement.cpu().numpy()

        sig_var = ndimage.variance(measurement_np)
        noise_var = sig_var / (10 ** (self.snr / 10))

        noisy = random_noise(
            measurement_np,
            mode=self.noise_type,
            clip=False,
            mean=self.noise_mean,
            var=noise_var,
        )
        return torch.tensor(np.array(noisy).astype(self.dtype)).to(measurement)


class CelebAPropagated(datasets.CelebA):
    def __init__(
        self,
        object_height,
        scene2mask,
        mask2sensor,
        sensor,
        attribute=None,
        psf_fp=None,
        single_psf=False,
        downsample_psf=None,
        output_dim=None,
        crop_psf=False,
        vflip=True,
        grayscale=True,
        device_conv=None,
        root="./data",
        split="train",
        scale=(1, 1),
        dtype=np.float32,
        dtype_out=torch.uint8,  # simulate quantization of sensor
        noise_type=None,
        snr=40,
        **kwargs,
    ):
        """
        attribute : str
            Attribute to use as label (from `CELEBA_ATTR`). Default is to return all.
        split : str
            One of {'train', 'valid', 'test', 'all'}.
        downsample_psf : float
            Downsample PSF to do smaller convolution.
        crop_psf : int
            To be used for lens PSF! How much to crop around peak of lens to extract the PSF.
        device_conv : str
            Convolution faster on GPU so can specify its device.
        """
        self.dtype = dtype
        self.dtype_out = dtype_out

        self.input_dim = np.array([218, 178])
        sensor_param = sensor_dict[sensor]
        sensor_size = sensor_param[SensorParam.SIZE]

        # -- load PSF
        if psf_fp is not None:
            psf = load_psf(fp=psf_fp, single_psf=single_psf, dtype=dtype)

            if crop_psf:
                # for compact support PSF like lens
                # -- keep full convolution
                self.conv_dim = np.array(psf.shape)

                # -- crop PSF around peak
                center = np.unravel_index(np.argmax(psf, axis=None), psf.shape)
                top = int(center[0] - crop_psf / 2)
                left = int(center[1] - crop_psf / 2)
                psf = psf[top : top + crop_psf, left : left + crop_psf]

            else:
                # for PSFs with large support, e.g. lensless
                if downsample_psf:
                    psf = resize(psf, 1 / downsample_psf, interpolation=cv2.INTER_CUBIC).astype(
                        dtype
                    )
                    if single_psf:
                        # cv2 drops the last dimension when it's 1..
                        psf = psf[:, :, np.newaxis]
                self.conv_dim = np.array(psf.shape)

            # reorder axis to [channels, width, height]
            if grayscale and not single_psf:
                psf = rgb2gray(psf).astype(dtype)
                psf = psf[np.newaxis, :, :]
                self.conv_dim[2] = 1
            else:
                psf = np.transpose(psf, (2, 0, 1))

            # cast as torch array
            psf = torch.tensor(psf, device=device_conv)
        else:
            # no PSF as we learn SLM. Simulate until mask
            # output dimensions is same as dimension (before) convolution
            psf = None
            assert output_dim is not None
            self.conv_dim = np.array(output_dim)

        self.psf = psf

        # processing steps
        # -- convert to tensor and flip image if need be
        transform_list = [np.array, transforms.ToTensor()]
        if vflip:
            transform_list.append(transforms.RandomVerticalFlip(p=1.0))

        # -- resize to convolution dimension and scale to desired height at object plane
        magnification = mask2sensor / scene2mask
        self.scene_dim = sensor_size / magnification
        object_height_pix = int(np.round(object_height / self.scene_dim[1] * self.conv_dim[1]))
        scaling = object_height_pix / self.input_dim[1]
        object_dim = (np.round(self.input_dim * scaling)).astype(int).tolist()
        transform_list.append(
            transforms.RandomResizedCrop(size=object_dim, ratio=(1, 1), scale=scale)
        )

        # -- pad rest with zeros
        padding = self.conv_dim[:2] - object_dim
        left = padding[1] // 2
        right = padding[1] - left
        top = padding[0] // 2
        bottom = padding[0] - top
        transform_list.append(transforms.Pad(padding=(left, top, right, bottom)))

        if grayscale:
            transform_list.append(transforms.Grayscale(num_output_channels=1))

        if self.psf is not None:
            self._transform_post = []

            # -- to do convolution on GPU (must faster)
            conv_op = RealFFTConvolve2D(
                self.psf, img_shape=np.roll(self.conv_dim, shift=1), device=device_conv
            )
            transform_list.append(conv_op)

            if noise_type:
                if noise_type == "poisson":
                    transform_list.append(AddPoissonNoise(snr))
                else:
                    if sensor == SensorOptions.RPI_HQ.value:
                        bit_depth = 12
                        noise_mean = RPI_HQ_CAMERA_BLACK_LEVEL / (2**bit_depth - 1)
                    else:
                        noise_mean = 0
                    transform_list.append(AddNoise(snr, noise_type, noise_mean, dtype))

            # -- resize to output dimension
            # more manageable to train on and perhaps don't need so many DOF
            if output_dim:
                transform_list.append(transforms.Resize(size=output_dim))

        transform = transforms.Compose(transform_list)
        super().__init__(root=root, split=split, download=False, transform=transform)

        # index extract label
        self.label = attribute
        if attribute is not None:
            self.label = self.attr_names.index(attribute)

    def __getitem__(self, index):

        res = super().__getitem__(index)
        img = res[0]

        # cast to uint8 as on sensor
        if img.max() > 1:
            img /= img.max()
        img *= 255
        img = img.to(dtype=self.dtype_out)

        # extract desired label
        if self.label is not None:
            label = res[1][self.label]
        else:
            label = res[1]

        return img, label


def get_object_height_pix(object_height, mask2sensor, scene2mask, sensor_dim, target_dim):
    """
    Determine height of object in pixel when it reaches the sensor.

    Parameters
    ----------
    object_height
    mask2sensor
    scene2mask
    sensor_dim
    target_dim

    Returns
    -------

    """
    magnification = mask2sensor / scene2mask
    scene_dim = sensor_dim / magnification
    return int(np.round(object_height / scene_dim[1] * target_dim[1]))


def get_dataset_stats(dataset, batch_size=100, num_workers=4):
    """
    Get mean and standard deviation.

    Example: https://kozodoi.me/python/deep%20learning/pytorch/tutorial/2021/03/08/image-mean-std.html

    """

    image_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    # placeholders
    psum = torch.tensor([0.0])
    psum_sq = torch.tensor([0.0])

    # loop through images
    for inputs in tqdm(image_loader):
        psum += inputs[0].sum(axis=[0, 2, 3])
        psum_sq += (inputs[0] ** 2).sum(axis=[0, 2, 3])

    # pixel count
    output_dim = dataset[0][0].shape
    count = len(dataset) * np.prod(output_dim)

    # mean and std
    total_mean = psum / count
    total_var = (psum_sq / count) - (total_mean**2)
    total_std = torch.sqrt(total_var)

    return total_mean.item(), total_std.item()


def simulate_propagated_dataset(
    dataset,
    down_psf,
    sensor,
    scene2mask,
    mask2sensor,
    object_height,
    crop_output,
    grayscale,
    single_psf,
    output_dir,
    device=None,
    device_conv=None,
    down_out=None,
    psf=None,
    print_progress=1000,
    n_files=None,
    output_dim=None,
    crop_psf=None,
    noise_type=None,
    snr=40,
    batch_size=100,
    n_workers=0,
    down="resize",
    random_shift=False,
    random_height=None,
    rotate=False,
    perspective=False,
):
    """
    psf : str
        File path to PSF. If not provided, simulating until PSF convolution,
        e.g. for learning SLM pattern.
    output_dir : str
        Where to save simulated dataset.
    """

    sensor_param = sensor_dict[sensor]
    if object_height is None:
        object_height = f"{random_height[0]}-{random_height[1]}"  # for creating directory name

    if psf is not None:
        psf_bn = os.path.basename(psf).split(".")[0]
        if down_out:
            output_dir = os.path.join(
                output_dir,
                f"{dataset}_{psf_bn}_down{int(down_out)}_height{object_height}_scene2mask{scene2mask}",
            )
        elif output_dim is not None:
            n_hidden = np.prod(output_dim)
            output_dir = os.path.join(
                output_dir,
                f"{dataset}_{psf_bn}_outdim{int(n_hidden)}_height{object_height}_scene2mask{scene2mask}",
            )

        if noise_type:
            output_dir += f"_{noise_type}{snr}"
        if crop_psf:
            output_dir += f"_croppsf{crop_psf}"
        if down_psf > 1:
            output_dir += f"_downpsf{down_psf}"
        if down != "resize":
            output_dir += f"_DS{down}"
        if not grayscale:
            output_dir += "_rgb"

        if output_dim is None:
            if down_out:
                output_dim = (sensor_param[SensorParam.SHAPE] * 1 / down_out).astype(int)
            else:
                output_dim = sensor_param[SensorParam.SHAPE]
    else:
        # no PSF as we learn SLM. Simulate until mask
        output_dir = os.path.join(
            output_dir, f"{dataset}_no_psf_down{int(down_psf)}_height{object_height}"
        )
        output_dim = (sensor_param[SensorParam.SHAPE] * 1 / down_psf).astype(int)
    output_dim = np.array(output_dim).tolist()

    if random_shift:
        output_dir += "_RandomShift"
    if rotate:
        output_dir += f"_RandomRotate{rotate}"
    if perspective:
        output_dir += f"_RandomPerspective{perspective}"
    if n_files:
        output_dir += f"_{n_files}files"

    if os.path.isdir(output_dir):
        print(f"\nDataset already exists: {output_dir}")
    else:
        print("\nSimulated dataset will be saved to :", output_dir)

    # initialize simulators
    args = {
        "psf_fp": psf,
        "downsample_psf": down_psf,
        "output_dim": output_dim,
        "scene2mask": scene2mask,
        "mask2sensor": mask2sensor,
        "sensor": sensor,
        "object_height": object_height,
        "device": device,
        "device_conv": device_conv,
        "crop_output": crop_output,
        "grayscale": grayscale,
        "vflip": False,
        "single_psf": single_psf,
        "crop_psf": crop_psf,
        "noise_type": noise_type,
        "snr": None if noise_type is None else snr,
        "down": down,
        "random_shift": random_shift,
        "rotate": rotate,
        "perspective": perspective,
        "random_height": random_height.tolist() if random_height is not None else None,
    }
    if dataset == "MNIST":
        ds_train = MNISTPropagated(**args, train=True)
        train_subdir = "train"
        ds_test = MNISTPropagated(**args, train=False)
    elif dataset == "celeba":
        ds_train = CelebAPropagated(**args, split="all")
        train_subdir = "all"
        ds_test = None

    ## loop over samples and save
    output_dir = plib.Path(output_dir)
    if not os.path.isdir(output_dir):
        output_dir.mkdir(exist_ok=True)

    metadata_fp = output_dir / "metadata.json"
    with open(metadata_fp, "w") as fp:
        json.dump(args, fp)

    # -- train set
    train_output = output_dir / train_subdir
    if not os.path.isdir(train_output):
        train_output.mkdir(exist_ok=True)

    train_labels = []
    start_time = time.time()

    # with DataLoader to parallelize
    train_loader = torch.utils.data.DataLoader(
        dataset=ds_train, batch_size=batch_size, shuffle=False, num_workers=n_workers
    )

    if n_files is None:
        n_files_train = len(ds_train)
    else:
        n_files_train = n_files

    n_png = len(list(train_output.glob("*.png")))
    if n_png < n_files_train:
        print(f"TRAIN SET : Augmenting {n_files_train - n_png} files...")

        n_batch_complete = n_png // batch_size
        reached_n_files = False

        if n_png:
            for i in range(n_png):
                label_fp = train_output / f"label{i}"
                train_labels.append(torch.load(label_fp))

        for batch_idx, (x, target) in enumerate(train_loader, start=n_batch_complete):

            for sample_idx, data in enumerate(x):

                i = batch_idx * batch_size + sample_idx

                if i == n_files_train:
                    reached_n_files = True
                    break

                output_fp = train_output / f"img{i}.png"
                label_fp = train_output / f"label{i}"

                if os.path.isfile(output_fp) and os.path.isfile(label_fp):
                    train_labels.append(torch.load(label_fp))
                else:
                    img_data = data.cpu().numpy().squeeze()

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
                    _label = target[sample_idx]
                    if hasattr(target, "__len__"):
                        _label = _label.numpy().tolist()
                    torch.save(_label, label_fp)
                    train_labels.append(_label)

                if i % print_progress == (print_progress - 1):
                    proc_time = time.time() - start_time
                    print(f"{i + 1} / {n_files_train} examples, {proc_time / 60} minutes")

            if reached_n_files:
                break

        with open(train_output / "labels.txt", "w") as f:
            for item in train_labels:
                f.write("%s\n" % item)

        proc_time = time.time() - start_time
        print(f"Processing time [m] : {proc_time/ 60}")
        print(f"Finished {train_subdir} set\n")

    # test set
    if ds_test:

        test_output = output_dir / "test"
        if not os.path.isdir(test_output):
            test_output.mkdir(exist_ok=True)

        test_labels = []
        start_time = time.time()

        # with DataLoader to parallelize
        test_loader = torch.utils.data.DataLoader(
            dataset=ds_test, batch_size=batch_size, shuffle=False, num_workers=n_workers
        )

        if n_files is None:
            n_files_test = len(ds_test)
        else:
            n_files_test = n_files

        n_complete_files = len(list(test_output.glob("*.png")))

        if n_complete_files < n_files_test:
            print(f"TEST SET : Augmenting {n_files_test - n_complete_files} files...")

            n_batch_complete = n_complete_files // batch_size
            reached_n_files = False

            if n_complete_files:
                for i in range(n_complete_files):
                    label_fp = test_output / f"label{i}"
                    test_labels.append(torch.load(label_fp))

            for batch_idx, (x, target) in enumerate(test_loader, start=n_batch_complete):

                for sample_idx, data in enumerate(x):

                    i = batch_idx * batch_size + sample_idx

                    if i == n_files_test:
                        reached_n_files = True
                        break

                    output_fp = test_output / f"img{i}.png"
                    label_fp = test_output / f"label{i}"

                    if os.path.isfile(output_fp) and os.path.isfile(label_fp):
                        test_labels.append(torch.load(label_fp))
                    else:
                        img_data = data.cpu().numpy().squeeze()

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
                        _label = target[sample_idx]
                        if hasattr(target, "__len__"):
                            _label = _label.numpy().tolist()
                        torch.save(_label, label_fp)
                        test_labels.append(_label)

                    if i % print_progress == (print_progress - 1):
                        proc_time = time.time() - start_time
                        print(f"{i + 1} / {n_files_test} examples, {proc_time / 60} minutes")

                if reached_n_files:
                    break

            with open(test_output / "labels.txt", "w") as f:
                for item in test_labels:
                    f.write("%s\n" % item)

            proc_time = time.time() - start_time
            print(f"Processing time [m] : {proc_time / 60}")
            print("Finished test set\n")

    return str(output_dir)
