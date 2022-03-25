from lensless.io import load_psf
from lensless.util import resize, rgb2gray
import cv2
import torch
from torchvision import transforms, datasets
from torch.utils.data import Dataset
import numpy as np
import os
import glob
from lenslessclass.util import RealFFTConvolve2D


# TODO : abstract parent class for DatasetPropagated
# TODO : take into account FOV and offset


class MNISTAugmented(Dataset):
    def __init__(self, path, train=True):
        self._path = path
        if train:
            self._subdir = os.path.join(path, "train")
        else:
            self._subdir = os.path.join(path, "test")
        self._n_files = len(glob.glob(os.path.join(self._subdir, "img*")))
        n_labels = len(glob.glob(os.path.join(self._subdir, "label*")))
        assert self._n_files == n_labels
        self._labels = []
        for i in range(n_labels):
            label_path = os.path.join(self._subdir, f"label{i}")
            self._labels.append(torch.load(label_path))

    def __getitem__(self, index):

        img_path = os.path.join(self._subdir, f"img{index}")
        img_tensor = torch.load(img_path)

        return img_tensor, self._labels[index]

    def __len__(self):
        return self._n_files


class MNISTPropagated(datasets.MNIST):
    def __init__(
        self,
        object_height,
        scene2mask,
        mask2sensor,
        sensor_size,
        psf_fp=None,
        single_psf=False,
        downsample_psf=None,  # default to psf dim, but could be too large
        output_dim=None,  # default to conv_dim, but could be too large
        crop_output=False,  # otherwise downsample output
        vflip=True,
        grayscale=True,
        fov=None,
        offset=None,
        device=None,
        root="./data",
        train=True,
        download=True,
        scale=(1, 1),
        dtype=np.float32,
        dtype_out=torch.uint8,  # simulate quantization of sensor
        **kwargs,
    ):

        self.dtype_out = dtype_out

        # -- load PSF
        if psf_fp is not None:
            psf = load_psf(fp=psf_fp, single_psf=single_psf, dtype=dtype)

            # resize
            if downsample_psf:
                psf = resize(psf, 1 / downsample_psf, interpolation=cv2.INTER_LINEAR).astype(dtype)
            self.conv_dim = np.array(psf.shape[:2])

            # cast as torch array
            if grayscale:
                psf = rgb2gray(psf).astype(dtype)
            else:
                psf = np.transpose(psf, (2, 0, 1))
            psf = torch.tensor(psf, device=device)
        else:
            psf = None
            assert output_dim is not None
            self.conv_dim = output_dim

        # -- convert to tensor and flip image if need be
        self.input_dim = np.array([28, 28])
        transform_list = [np.array, transforms.ToTensor()]
        if vflip:
            transform_list.append(transforms.RandomVerticalFlip(p=1.0))

        # -- resize to convolution dimension and scale to desired height at object plane
        magnification = mask2sensor / scene2mask
        self.scene_dim = sensor_size / magnification
        if psf is not None:
            object_height_pix = int(np.round(object_height / self.scene_dim[1] * self.conv_dim[1]))
        else:
            object_height_pix = int(np.round(object_height / self.scene_dim[1] * output_dim[1]))
        scaling = object_height_pix / self.input_dim[1]
        object_dim = (np.round(self.input_dim * scaling)).astype(int).tolist()
        transform_list.append(
            transforms.RandomResizedCrop(size=object_dim, ratio=(1, 1), scale=scale)
        )

        # pad rest with zeros
        padding = self.conv_dim - object_dim
        left = padding[1] // 2
        right = padding[1] - left
        top = padding[0] // 2
        bottom = padding[0] - top
        transform_list.append(transforms.Pad(padding=(left, top, right, bottom)))

        if not grayscale:
            transform_list.append(transforms.Lambda(lambda x: x.repeat(3, 1, 1)))

        transform = transforms.Compose(transform_list)

        # -- to do convolution on GPU (must faster)
        self._transform_post = []
        if psf is not None:
            conv_op = RealFFTConvolve2D(psf)
            self._transform_post.append(conv_op)

        if crop_output:
            # remove previous padding
            center = (psf == torch.max(psf)).nonzero().cpu().numpy()[0]

            def crop(img):
                top = int(center[0] - object_dim[0] / 2)
                left = int(center[1] - object_dim[1] / 2)
                return transforms.functional.crop(
                    img, top=top, left=left, height=object_dim[0], width=object_dim[1]
                )

            self._transform_post.append(crop)

        # -- resize to output dimension
        # more manageable to train on and perhaps don't need so many DOF
        if output_dim:
            self._transform_post.append(transforms.Resize(size=output_dim))

        self._transform_post = transforms.Compose(self._transform_post)

        self.device = device
        super().__init__(root=root, train=train, download=download, transform=transform)

    def __getitem__(self, index):

        res = super().__getitem__(index)
        if self.device:
            img = res[0].to(device=self.device)
        else:
            img = res[0]

        img = self._transform_post(img)

        # clip to 1
        if img.max() > 1:
            # todo: clip instead?
            img /= img.max()
        img *= 255
        img = img.to(dtype=self.dtype_out)

        return img, res[1]


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
