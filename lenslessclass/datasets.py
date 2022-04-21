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


# TODO : abstract parent class for DatasetPropagated
# TODO : take into account FOV and offset


class MNISTAugmented(Dataset):
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
        # self.output_dim = np.array(img.size)

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
        sensor_size,
        psf_fp=None,
        single_psf=False,
        downsample_psf=None,  # default to psf dim, but could be too large
        output_dim=None,  # default to conv_dim, but could be too large
        crop_output=False,  # otherwise downsample output
        vflip=True,
        grayscale=True,
        device=None,
        root="./data",
        train=True,
        download=True,
        scale=(1, 1),
        dtype=np.float32,
        dtype_out=torch.uint8,  # simulate quantization of sensor
        fov=None,
        offset=None,
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
            # output dimensions is same as dimension of convolution
            psf = None
            assert output_dim is not None
            self.conv_dim = np.array(output_dim)

        # -- convert to tensor and flip image if need be
        self.input_dim = np.array([28, 28])
        transform_list = [np.array, transforms.ToTensor()]
        if vflip:
            transform_list.append(transforms.RandomVerticalFlip(p=1.0))

        # -- resize to convolution dimension and scale to desired height at object plane
        magnification = mask2sensor / scene2mask
        self.scene_dim = sensor_size / magnification
        object_height_pix = int(np.round(object_height / self.scene_dim[1] * self.conv_dim[1]))
        # if psf is not None:
        #     object_height_pix = int(np.round(object_height / self.scene_dim[1] * self.conv_dim[1]))
        # else:
        #     object_height_pix = int(np.round(object_height / self.scene_dim[1] * output_dim[1]))
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
        self._transform_post = None
        if psf is not None:
            self._transform_post = []

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

        if self._transform_post:
            img = self._transform_post(img)

        # cast to uint8 as on sensor
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


def simulate_propagated_dataset(
    psf,
    down_psf,
    sensor,
    down_out,
    scene2mask,
    mask2sensor,
    object_height,
    device,
    crop_output,
    grayscale,
    single_psf,
    output_dir,
    batch=1000,
    n_files=None,
    return_output_dir=False,
):
    """
    output_dir : str
        Where to save simulated dataset.
    """

    psf_bn = os.path.basename(psf).split(".")[0]
    output_dir = os.path.join(output_dir, f"MNIST_{psf_bn}_down{int(down_out)}")
    if n_files:
        output_dir += f"_{n_files}files"
    if not grayscale:
        output_dir += "_rgb"

    if return_output_dir:
        return output_dir

    print("\nSimulated dataset will be saved to :", output_dir)

    sensor_param = sensor_dict[sensor]
    if down_out:
        output_dim = tuple((sensor_param[SensorParam.SHAPE] * 1 / down_out).astype(int))

    ds_train = MNISTPropagated(
        psf_fp=psf,
        downsample_psf=down_psf,
        output_dim=output_dim,
        scene2mask=scene2mask,
        mask2sensor=mask2sensor,
        sensor_size=sensor_param[SensorParam.SIZE],
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
        downsample_psf=down_psf,
        output_dim=output_dim,
        scene2mask=scene2mask,
        mask2sensor=mask2sensor,
        sensor_size=sensor_param[SensorParam.SIZE],
        object_height=object_height,
        device=device,
        crop_output=crop_output,
        grayscale=grayscale,
        vflip=False,
        train=False,
        single_psf=single_psf,
    )

    ## loop over samples and save
    output_dir = plib.Path(output_dir)
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
        if os.path.isfile(output_fp) and os.path.isfile(label_fp):
            train_labels.append(torch.load(label_fp))
        else:
            data = ds_train[i]
            img_data = data[0].cpu().clone().numpy().squeeze()

            if img_data.dtype == np.uint8:
                # save as viewable images
                if len(img_data) == 3:
                    # RGB
                    img_data = img_data.transpose(1, 2, 0)
                im = Image.fromarray(img_data)
                im.save(output_fp)
            else:
                # save as float data
                np.save(output_fp, img_data)

            # save label
            torch.save(data[1], label_fp)
            train_labels.append(data[1])

        if i % batch == (batch - 1):
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
            img_data = data[0].cpu().clone().numpy().squeeze()

            if img_data.dtype == np.uint8:
                # save as viewable images
                if len(img_data) == 3:
                    # RGB
                    img_data = img_data.transpose(1, 2, 0)
                im = Image.fromarray(img_data)
                im.save(output_fp)

            else:
                # save as flaot data
                np.save(output_fp, img_data)

            # save label
            torch.save(data[1], label_fp)
            test_labels.append(data[1])

        if i % batch == (batch - 1):
            proc_time = time.time() - start_time
            print(f"{i + 1} / {len(ds_test)} examples, {proc_time / 60} minutes")

    with open(test_output / "labels.txt", "w") as f:
        for item in test_labels:
            f.write("%s\n" % item)

    proc_time = time.time() - start_time
    print(f"Processing time [m] : {proc_time / 60}")
    print("Finished test set")

    return str(output_dir)
