from torch import nn
import torch.nn.functional as F
import torch
import warnings
from torchvision import transforms
from torchvision.transforms.functional import rgb_to_grayscale, resize
import numpy as np
from waveprop.slm import get_active_pixel_dim, get_slm_mask
from waveprop.pytorch_util import fftconvolve
from waveprop.spherical import spherical_prop
from waveprop.color import ColorSystem
from waveprop.rs import angular_spectrum
from lensless.constants import RPI_HQ_CAMERA_BLACK_LEVEL
from skimage.util.noise import random_noise
from scipy import ndimage
from lenslessclass.util import AddPoissonNoise
from waveprop.devices import SLMOptions, SensorOptions, slm_dict, sensor_dict, SensorParam
from collections import OrderedDict
import cv2
from lenslessclass.vgg import VGG


class MultiClassLogistic(nn.Module):
    """
    Example: https://gist.github.com/xmfbit/b27cdbff68870418bdb8cefa86a2d558
    """

    def __init__(self, input_shape, multi_gpu=False, n_class=10):
        super(MultiClassLogistic, self).__init__()
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(int(np.prod(input_shape)), n_class)
        self.decision = nn.Softmax(dim=1)

        if multi_gpu:
            self.linear1 = nn.DataParallel(self.linear1, device_ids=multi_gpu)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear1(x)
        return self.decision(x)

    def name(self):
        return "MultiClassLogistic"


class BinaryLogistic(nn.Module):
    def __init__(self, input_shape):
        super(BinaryLogistic, self).__init__()
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(int(np.prod(input_shape)), 1)
        self.decision = nn.Sigmoid()

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear1(x)
        return self.decision(x)

    def name(self):
        return "BinaryLogistic"


class SingleHidden(nn.Module):
    """
    Example: https://gist.github.com/xmfbit/b27cdbff68870418bdb8cefa86a2d558
    """

    def __init__(self, input_shape, hidden_dim, n_class, bn=True, dropout=None):
        assert isinstance(hidden_dim, int)
        self.hidden_dim = hidden_dim
        super(SingleHidden, self).__init__()
        self.flatten = nn.Flatten()
        if bn:
            self.linear1 = nn.Linear(int(np.prod(input_shape)), hidden_dim, bias=False)
            self.bn1 = nn.BatchNorm1d(hidden_dim)  # ADDED to be consistent with SLM
        else:
            self.linear1 = nn.Linear(int(np.prod(input_shape)), hidden_dim, bias=True)
            self.bn1 = None
        self.activation1 = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, n_class)

        if dropout:
            self.dropout_rate = dropout
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout_rate = None
            self.dropout = None

        if n_class > 1:
            self.decision = nn.Softmax(dim=1)
        else:
            self.decision = nn.Sigmoid()

    def forward(self, x):
        x = self.flatten(x)
        if self.bn1 is not None:
            x = self.activation1(self.bn1(self.linear1(x)))  # ADDED to be consistent with SLM
        else:
            x = self.activation1(self.linear1(x))
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.linear2(x)
        logits = self.decision(x)
        return logits

    def name(self):
        model_name = f"SingleHidden{self.hidden_dim}"
        if self.dropout:
            model_name += f"drop{self.dropout_rate}"
        return model_name


class TwoHidden(nn.Module):
    """
    Example: https://gist.github.com/xmfbit/b27cdbff68870418bdb8cefa86a2d558
    """

    def __init__(self, input_shape, hidden_dim, hidden_dim_2, n_class):
        assert isinstance(hidden_dim, int)
        self.hidden_dim = hidden_dim
        self.hidden_dim_2 = hidden_dim_2
        super(TwoHidden, self).__init__()
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(int(np.prod(input_shape)), hidden_dim, bias=False)
        self.bn1 = nn.BatchNorm1d(hidden_dim)  # ADDED to be consistent with SLM
        self.linear2 = nn.Linear(hidden_dim, hidden_dim_2, bias=False)
        self.bn2 = nn.BatchNorm1d(hidden_dim_2)  # ADDED to be consistent with SLM
        self.activation = nn.ReLU()
        self.linear3 = nn.Linear(hidden_dim_2, n_class)
        if n_class > 1:
            self.decision = nn.Softmax(dim=1)
        else:
            self.decision = nn.Sigmoid()

    def forward(self, x):
        x = self.flatten(x)
        x = self.activation(self.bn1(self.linear1(x)))  # ADDED to be consistent with SLM
        x = self.activation(self.bn2(self.linear2(x)))
        x = self.linear3(x)
        logits = self.decision(x)
        return logits

    def name(self):
        return f"TwoHidden{self.hidden_dim}_{self.hidden_dim_2}"


def conv_output_dim(input_dim, kernel, padding=0, stride=1, pooling=1):
    """
    https://stackoverflow.com/questions/53580088/calculate-the-output-size-in-convolution-layer
    """
    if isinstance(kernel, int):
        kernel = [kernel, kernel]
    if isinstance(padding, int):
        padding = [padding, padding]
    if isinstance(stride, int):
        stride = [stride, stride]
    if isinstance(pooling, int):
        pooling = [pooling, pooling]
    output_dim = (np.array(input_dim) - np.array(kernel) + 2 * np.array(padding)) / stride + 1
    return (output_dim // pooling).astype(np.int)


class CNNLite(nn.Module):
    def __init__(
        self,
        input_shape,
        n_class,
        n_kern=10,
        bn=True,
        kernel_size=3,
        pool=2,
        hidden=800,
        dropout=None,
    ):
        """
        Single convolution layer -> flatten > fully connected > softmax
        """

        assert len(input_shape) == 2

        super().__init__()
        self.n_kern = n_kern
        self.kernel_size = kernel_size
        self.pool = pool
        self.hidden = hidden

        self.conv = nn.Conv2d(1, self.n_kern, self.kernel_size)
        output_dim = conv_output_dim(
            input_shape,
            kernel=kernel_size,
            pooling=self.pool,
        )

        if hidden:
            if bn:
                self.fc1 = nn.Linear(self.n_kern * np.prod(output_dim), self.hidden, bias=False)
                self.bn1 = nn.BatchNorm1d(self.hidden)
            else:
                self.fc1 = nn.Linear(self.n_kern * np.prod(output_dim), self.hidden)
                self.bn1 = None
            self.fc2 = nn.Linear(self.hidden, n_class)
        else:
            self.fc1 = nn.Linear(self.n_kern * np.prod(output_dim), n_class)
            # if bn:
            #     self.fc1 = nn.Linear(self.n_kern * np.prod(output_dim), n_class, bias=False)
            #     self.bn1 = nn.BatchNorm1d(self.hidden)
            # else:
            #     self.fc1 = nn.Linear(self.n_kern * np.prod(output_dim), n_class)
            #     self.bn1 = None

        if dropout:
            self.dropout_rate = dropout
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout_rate = None
            self.dropout = None

        if n_class > 1:
            self.decision = nn.Softmax(dim=1)
        else:
            self.decision = nn.Sigmoid()

    def forward(self, x):

        # Max pooling
        x = F.relu(self.conv(x))
        if self.pool > 1:
            x = F.max_pool2d(x, (self.pool, self.pool))
        x = torch.flatten(x, 1)  # flatten all dimensions except the batch dimension
        if self.dropout:
            x = self.dropout(x)

        if self.hidden:
            if self.bn1 is not None:
                x = F.relu(self.bn1(self.fc1(x)))
            else:
                x = F.relu(self.fc1(x))
            if self.dropout:
                x = self.dropout(x)
            x = self.fc2(x)
        else:
            x = self.fc1(x)
        return self.decision(x)

    def name(self):
        model_name = f"CNNLite{self.n_kern}_FC{self.hidden}"
        if self.dropout:
            model_name = model_name + f"_dropout{self.dropout_rate}"
        return model_name


class CNN(nn.Module):
    """
    Example: https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html#define-the-network
    Example: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#define-a-convolutional-neural-network
    """

    def __init__(self, input_shape, n_class, n_kern=6, bn=True, pool=2, kernel_size=3):
        super().__init__()
        self.n_kern = n_kern
        self.n_kern2 = 20
        self.pool = pool
        kernel_size = kernel_size
        # 1 input image channel, 6 output channels, kernelxkernel square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, self.n_kern, kernel_size)
        self.conv2 = nn.Conv2d(self.n_kern, self.n_kern2, kernel_size)
        # an affine operation: y = Wx + b
        output_dim = conv_output_dim(
            conv_output_dim(input_shape, kernel=kernel_size, pooling=self.pool),
            kernel=kernel_size,
            pooling=self.pool,
        )
        if bn:
            self.fc1 = nn.Linear(self.n_kern2 * np.prod(output_dim), 120, bias=False)
            self.bn1 = nn.BatchNorm1d(120)
        else:
            self.fc1 = nn.Linear(self.n_kern2 * np.prod(output_dim), 120)
            self.bn1 = None
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, n_class)
        if n_class > 1:
            self.decision = nn.Softmax(dim=1)
        else:
            self.decision = nn.Sigmoid()

    def forward(self, x):

        # Max pooling
        x = F.relu(self.conv1(x))
        if self.pool > 1:
            x = F.max_pool2d(x, (self.pool, self.pool))
        x = F.relu(self.conv2(x))
        if self.pool > 1:
            x = F.max_pool2d(x, (self.pool, self.pool))
        x = torch.flatten(x, 1)  # flatten all dimensions except the batch dimension
        if self.bn1 is not None:
            x = F.relu(self.bn1(self.fc1(x)))
        else:
            x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        logits = self.decision(x)
        return logits

    def name(self):
        return f"CNN{self.n_kern}_{self.n_kern2}"


class DeepBig(nn.Module):
    """
    Default to 6-layer from this paper: https://arxiv.org/abs/1003.0358
    """

    def __init__(self, input_shape, n_class, hidden_dim=None, dropout=None):
        super(DeepBig, self).__init__()
        self.flatten = nn.Flatten()

        if hidden_dim is None:
            # hidden_dim = [2500, 2000, 1500, 1000, 500]
            hidden_dim = [1122, 800]

        layer_dim = [int(np.prod(input_shape))] + hidden_dim
        self.n_layers = len(layer_dim)

        layers = []
        for i in range(len(layer_dim) - 1):
            layers += [
                (f"linear{i+1}", nn.Linear(layer_dim[i], layer_dim[i + 1], bias=False)),
                (f"bn{i+1}", nn.BatchNorm1d(layer_dim[i + 1])),
                (f"relu{i+1}", nn.ReLU()),
            ]
            if dropout is not None:
                layers += [
                    (f"dropout{i+1}", nn.Dropout(p=dropout)),
                ]
        layers += [(f"linear{len(layer_dim)}", nn.Linear(layer_dim[-1], n_class))]
        self.linear_layers = nn.Sequential(OrderedDict(layers))
        if n_class > 1:
            self.decision = nn.Softmax(dim=1)
        else:
            self.decision = nn.Sigmoid()

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear_layers(x)
        logits = self.decision(x)
        return logits

    def name(self):
        return f"DeepBig{self.n_layers}"


class SLMMultiClassLogistic(nn.Module):
    """ """

    def __init__(
        self,
        input_shape,
        slm_config,
        sensor_config,
        crop_fact,
        scene2mask,
        mask2sensor,
        n_class,
        target_dim=None,  # try to get this while keeping aspect ratio of sensor
        down="resize",
        device="cpu",
        dtype=torch.float32,
        deadspace=True,
        first_color=0,
        grayscale=True,
        device_mask_creation=None,
        multi_gpu=False,  # or pass list of devices
        sensor_activation=None,
        dropout=None,
        noise_type=None,
        snr=40,
        return_measurement=False,
        hidden=None,
        hidden2=None,
        output_dim=None,
        requires_grad=True,
        n_kern=None,  # flag to use CNN
        n_slm_mask=1,  # number of SLM masks to optimize, would be time-multiplexed
        kernel_size=3,
        pool=2,
        cnn_lite=None,
        bn=True,
        vgg=None,
        **kwargs,
    ):
        """
        grayscale : whether input is grayscale, can then simplify to just grayscale PSF

        """
        super(SLMMultiClassLogistic, self).__init__()

        assert n_class > 0
        if dtype == torch.float32:
            self.ctype = torch.complex64
        elif dtype == torch.float64:
            self.ctype = torch.complex128
        else:
            raise ValueError(f"Unsupported data type : {dtype}")

        if len(input_shape) == 2:
            input_shape = [1] + list(input_shape)

        # store configuration
        self.input_shape = np.array(input_shape)
        self.hidden = hidden
        self.hidden2 = hidden2
        self.slm_config = slm_config
        self.sensor_config = sensor_config
        self.crop_fact = crop_fact
        self.scene2mask = scene2mask
        self.mask2sensor = mask2sensor
        self.deadspace = deadspace
        self.first_color = first_color
        self.grayscale = grayscale
        self.device = device
        if device_mask_creation is None:
            device_mask_creation = device
        else:
            self.device_mask_creation = device_mask_creation
        self.dtype = dtype
        self.target_dim = target_dim
        self.return_measurement = return_measurement
        self.requires_grad = requires_grad  # for SLM vals
        self.n_slm_mask = n_slm_mask

        # adding noise
        if noise_type:
            if noise_type == "poisson":
                add_noise = AddPoissonNoise(snr)

            else:
                # TODO : hardcoded for Raspberry Pi HQ sensor
                bit_depth = 12
                noise_mean = RPI_HQ_CAMERA_BLACK_LEVEL / (2**bit_depth - 1)

                def add_noise(measurement):

                    # normalize as mean is normalized to max value 1
                    with torch.no_grad():
                        max_vals = torch.max(torch.flatten(measurement, start_dim=1), dim=1)[0]
                        max_vals = max_vals.unsqueeze(1).unsqueeze(1).unsqueeze(1)
                    measurement /= max_vals

                    # compute noise for each image
                    measurement_np = measurement.clone().cpu().detach().numpy()
                    noise = []
                    for _measurement in measurement_np:

                        sig_var = ndimage.variance(_measurement)
                        noise_var = sig_var / (10 ** (snr / 10))
                        noise.append(
                            random_noise(
                                _measurement,
                                mode=noise_type,
                                clip=False,
                                mean=noise_mean,
                                var=noise_var,
                            )
                            - _measurement
                        )

                    noise = torch.tensor(np.array(noise).astype(np.float32)).to(device)

                    return measurement + noise

            self.add_noise = add_noise
        else:
            self.add_noise = None

        # -- downsampling
        self.downsample = None

        if self.target_dim is not None or output_dim is not None:
            if output_dim is None:
                sensor_size = sensor_config[SensorParam.SHAPE]
                w = np.sqrt(np.prod(target_dim) * sensor_size[1] / sensor_size[0])
                h = sensor_size[0] / sensor_size[1] * w
                self.output_dim = np.array([int(h), int(w)])
            else:
                self.output_dim = np.array(output_dim)

            if not grayscale:
                self.output_dim = np.r_[self.output_dim, 3]

            if down == "resize":

                self.downsample = transforms.Resize(size=self.output_dim[:2].tolist())

            elif down == "max" or down == "avg":
                n_embedding = np.prod(self.output_dim)

                # determine filter size, stride, and padding: https://androidkt.com/calculate-output-size-convolutional-pooling-layers-cnn/
                k = int(np.ceil(np.sqrt(np.prod(self.input_shape[1:]) / n_embedding)))
                p = np.roots(
                    [
                        4,
                        2 * np.sum(self.input_shape),
                        np.prod(self.input_shape[1:]) - k**2 * n_embedding,
                    ]
                )
                p = max(int(np.max(p)), 0) + 1
                if down == "max":
                    self.downsample = nn.MaxPool2d(kernel_size=k, stride=k, padding=p)
                else:
                    self.downsample = nn.AvgPool2d(kernel_size=k, stride=k, padding=p)
                pooling_outdim = ((self.input_shape[1:] - k + 2 * p) / k + 1).astype(int)
                assert np.array_equal(self.output_dim[:2], pooling_outdim)
            else:
                raise ValueError("Invalid downsampling approach.")
        else:
            self.output_dim = self.input_shape

        self.n_hidden = int(np.prod(self.output_dim))

        # -- decision network after sensor
        self.sensor_activation = sensor_activation
        self.flatten = nn.Flatten()
        if dropout is not None:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        self.n_kern = n_kern
        self.cnn_lite = cnn_lite
        self.classifier = None
        self.vgg = vgg
        if vgg:
            self.classifier = VGG(vgg, input_shape=np.roll(self.output_dim, shift=1))
        elif cnn_lite:
            self.classifier = CNNLite(
                input_shape=self.output_dim,
                n_kern=cnn_lite,
                kernel_size=kernel_size,
                n_class=n_class,
                hidden=hidden,
                pool=pool,
                dropout=dropout,
            )
        elif n_kern:
            self.classifier = CNN(
                input_shape=self.output_dim,
                n_class=n_class,
                n_kern=n_kern,
                bn=bn,
                kernel_size=kernel_size,
                pool=pool,
            )
        elif self.hidden2 is not None:
            assert self.hidden is not None
            if bn:
                self.linear_layers = nn.Sequential(
                    OrderedDict(
                        [
                            (
                                "linear1",
                                nn.Linear(self.n_slm_mask * self.n_hidden, self.hidden, bias=False),
                            ),
                            ("bn", nn.BatchNorm1d(self.hidden)),
                            ("relu1", nn.ReLU()),
                            ("linear2", nn.Linear(self.hidden, self.hidden2, bias=False)),
                            ("bn2", nn.BatchNorm1d(self.hidden2)),
                            ("relu2", nn.ReLU()),
                            ("linear3", nn.Linear(self.hidden2, n_class)),
                        ]
                    )
                )
            else:
                self.linear_layers = nn.Sequential(
                    OrderedDict(
                        [
                            (
                                "linear1",
                                nn.Linear(self.n_slm_mask * self.n_hidden, self.hidden, bias=True),
                            ),
                            ("relu1", nn.ReLU()),
                            ("linear2", nn.Linear(self.hidden, self.hidden2, bias=True)),
                            ("relu2", nn.ReLU()),
                            ("linear3", nn.Linear(self.hidden2, n_class)),
                        ]
                    )
                )
        else:
            self.bn = None
            if self.hidden is not None and self.hidden:
                if bn:
                    self.linear1 = nn.Linear(
                        self.n_slm_mask * self.n_hidden, self.hidden, bias=False
                    )
                    self.bn = nn.BatchNorm1d(self.hidden)
                else:
                    self.linear1 = nn.Linear(
                        self.n_slm_mask * self.n_hidden, self.hidden, bias=True
                    )
                self.linear2 = nn.Linear(self.hidden, n_class)
            else:
                self.linear1 = nn.Linear(self.n_slm_mask * self.n_hidden, n_class)
                self.linear2 = None
            self.linear_layers = None

        if n_class > 1:
            self.decision = nn.Softmax(dim=1)
        else:
            self.decision = nn.Sigmoid()

        # -- determine number of active SLM pixels
        overlapping_mask_size, overlapping_mask_dim, n_active_slm_pixels = get_active_pixel_dim(
            sensor_config=sensor_config,
            sensor_crop=crop_fact,
            slm_config=slm_config,
        )

        # -- initialize SLM values, set as parameter to optimize
        # TODO : need to make internal changes to get_slm_mask for no deadspace
        rand_func = torch.rand  # between [0, 1]
        # rand_func = torch.randn
        if deadspace:
            if self.n_slm_mask == 1:
                self.slm_vals = rand_func(
                    *n_active_slm_pixels,
                    dtype=dtype,
                    device=self.device,
                    requires_grad=self.requires_grad,
                )
                if self.requires_grad:
                    self.slm_vals = nn.Parameter(self.slm_vals)
            else:
                self.slm_vals = [
                    rand_func(
                        *n_active_slm_pixels,
                        dtype=dtype,
                        device=self.device,
                        requires_grad=self.requires_grad,
                    )
                    for _ in range(self.n_slm_mask)
                ]
                if self.requires_grad:
                    self.slm_vals = nn.ParameterList(
                        [nn.Parameter(self.slm_vals[i]) for i in range(self.n_slm_mask)]
                    )

        else:
            if self.requires_grad:
                self.slm_vals = nn.Parameter(
                    rand_func(
                        *overlapping_mask_dim,
                        dtype=dtype,
                        device=self.device,
                        requires_grad=self.requires_grad,
                    )
                )
            else:
                self.slm_vals = rand_func(
                    *overlapping_mask_dim,
                    dtype=dtype,
                    device=self.device,
                    requires_grad=self.requires_grad,
                )

        # -- normalize after PSF
        if self.grayscale:
            self.conv_bn = nn.BatchNorm2d(self.n_slm_mask)
        else:
            self.conv_bn = nn.BatchNorm2d(self.n_slm_mask * 3)

        # -- initialize PSF from SLM values and pre-compute constants
        # object to mask (modeled with spherical propagation which can be pre-computed)
        self.color_system = ColorSystem.rgb()
        self.d1 = np.array(overlapping_mask_size) / self.input_shape[1:]
        self.spherical_wavefront = spherical_prop(
            in_shape=self.input_shape[1:],
            d1=self.d1,
            wv=self.color_system.wv,
            dz=self.scene2mask,
            return_psf=True,
            is_torch=True,
            device=self.device,
            dtype=self.dtype,
        )

        self._psf = None
        self._H = None  # pre-compute free space propagation kernel
        self._H_exp = None

        slm_vals = self.get_slm_vals()
        self.compute_intensity_psf(slm_vals=slm_vals)

        # -- parallelize across GPUs
        if multi_gpu:
            self.downsample = nn.DataParallel(self.downsample, device_ids=multi_gpu)
            if self.classifier is not None:
                # CNN classifier
                self.classifier = nn.DataParallel(self.classifier, device_ids=multi_gpu)
            elif self.linear_layers is not None:
                self.linear_layers = nn.DataParallel(self.linear_layers, device_ids=multi_gpu)
            else:
                self.conv_bn = nn.DataParallel(self.conv_bn, device_ids=multi_gpu)
                self.linear1 = nn.DataParallel(self.linear1, device_ids=multi_gpu)
                if self.linear2:
                    self.linear2 = nn.DataParallel(self.linear2, device_ids=multi_gpu)
                    if self.bn is not None:
                        self.bn = nn.DataParallel(self.bn, device_ids=multi_gpu)

    def get_slm_vals(self):
        """
        Apply any pre-processing to learned SLM values, e.g. clamping within [0, 1].

        TODO quantize / use look up table
        """

        # TRACKING GRADIENT WITH CLAMPING
        if self.n_slm_mask == 1:
            # slm_vals = self.slm_vals.sigmoid()
            slm_vals = self.slm_vals.clamp(min=0, max=1)  # found clamp to be better
        else:
            # slm_vals = [self.slm_vals[i].sigmoid() for i in range(self.n_slm_mask)]
            slm_vals = [self.slm_vals[i].clamp(min=0, max=1) for i in range(self.n_slm_mask)]

        return slm_vals

        # with torch.no_grad():
        #     if self.n_slm_mask == 1:
        #         self.slm_vals.clamp_(min=0, max=1)
        #     else:
        #         slm_vals = [self.slm_vals[i].clamp_(min=0, max=1) for i in range(self.n_slm_mask)]

        # return self.slm_vals

    def set_slm_vals(self, slm_vals):
        """
        only works if requires_grad = False
        """

        np.testing.assert_array_equal(slm_vals.shape, self.slm_vals.shape)
        self.slm_vals = slm_vals

        # recompute intensity PSF
        self.compute_intensity_psf()

    def get_psf(self, numpy=False):
        slm_vals = self.get_slm_vals()
        self.compute_intensity_psf(slm_vals=slm_vals)

        if numpy:
            if self.n_slm_mask == 1:
                return self._psf.cpu().detach().numpy().squeeze()
            else:
                return [_psf.cpu().detach().numpy().squeeze() for _psf in self._psf]
        else:
            return self._psf

    def set_mask2sensor(self, mask2sensor):
        self.mask2sensor = mask2sensor

        # recompute intensity PSF
        self.compute_intensity_psf()

    def save_psf(self, fp, bit_depth=8):
        psf = self.get_psf(numpy=True)
        psf = np.transpose(psf, (1, 2, 0))
        psf /= psf.max()

        # save as int
        psf *= 2**bit_depth - 1
        if bit_depth <= 8:
            psf = psf.astype(dtype=np.uint8)
        else:
            psf = psf.astype(dtype=np.uint16)
        cv2.imwrite(fp, cv2.cvtColor(psf, cv2.COLOR_RGB2BGR))

    def forward(self, x):

        if x.min() < 0:
            warnings.warn("Got negative data. Shift to non-negative.")
            x -= x.min()

        # compute intensity PSF from SLM values
        slm_vals = self.get_slm_vals()  # apply any (physical) pre-processing
        self.compute_intensity_psf(slm_vals=slm_vals)

        if self.n_slm_mask > 1:
            # add dimension for time-multiplexed measurements
            x = x.unsqueeze(1)

        # convolve with PSF
        x = fftconvolve(x, self._psf, axes=(-2, -1))

        if self.n_slm_mask > 1:
            # consider time-multiplexed as more channels
            x = x.flatten(1, 2)

        if self.downsample is not None:
            x = self.downsample(x)

        if self.add_noise is not None:
            x = self.add_noise(x)

        # make sure non-negative
        x = torch.clip(x, min=0)

        if self.return_measurement:
            return x

        # normalize after PSF
        x = self.conv_bn(x)

        if self.sensor_activation is not None:
            x = self.sensor_activation(x)

        if self.vgg is not None:
            # TODO : more elegant approach to make square input to VGG??
            x = resize(x, size=(32, 32))

        # -- digital decision network after sensor
        if self.classifier is not None:
            assert self.n_slm_mask == 1, "Not supported for multiple masks"
            logits = self.classifier(x)
        else:
            x = self.flatten(x)
            if self.linear_layers is not None:
                x = self.linear_layers(x)
            else:
                x = self.linear1(x)
                if self.dropout is not None:
                    x = self.dropout(x)
                if self.hidden:
                    if self.bn is not None:
                        x = self.bn(x)
                    x = self.sensor_activation(x)
                    x = self.linear2(x)
            logits = self.decision(x)
        return logits

    def compute_intensity_psf(self, slm_vals=None):

        if slm_vals is None:
            slm_vals = self.get_slm_vals()

        assert slm_vals.max() <= 1
        assert slm_vals.min() >= 0

        if self.n_slm_mask == 1:
            # TODO : backward compatability but can make consistent with multiple masks

            # -- get SLM mask, i.e. deadspace modeling, quantization (todo), non-linearities (todo), etc
            mask = get_slm_mask(
                slm_vals=slm_vals.to(self.device_mask_creation),
                slm_config=self.slm_config,
                sensor_config=self.sensor_config,
                crop_fact=self.crop_fact,
                target_dim=self.input_shape[1:],
                deadspace=self.deadspace,
                device=self.device_mask_creation,
                dtype=self.dtype,
                requires_grad=self.requires_grad,
            )
            if self.device != self.device_mask_creation:
                mask = mask.to(self.device)

            # apply mask
            u_in = mask * self.spherical_wavefront

            # mask to sensor
            if self._H is None:
                # precompute at very start, not dependent on input pattern, just its shape
                # TODO : benchmark how much it actually saves
                self._H = torch.zeros(
                    [self.color_system.n_wavelength] + list(self.input_shape[1:] * 2),
                    dtype=self.ctype,
                    device=self.device,
                )
                for i in range(self.color_system.n_wavelength):
                    self._H[i] = angular_spectrum(
                        u_in=u_in[i],
                        wv=self.color_system.wv[i],
                        d1=self.d1,
                        dz=self.mask2sensor,
                        dtype=self.dtype,
                        device=self.device,
                        return_H=True,
                    )

            psfs = torch.zeros(u_in.shape, dtype=self.ctype, device=self.device)
            for i in range(self.color_system.n_wavelength):
                psfs[i], _, _ = angular_spectrum(
                    u_in=u_in[i],
                    wv=self.color_system.wv[i],
                    d1=self.d1,
                    dz=self.mask2sensor,
                    dtype=self.dtype,
                    device=self.device,
                    H=self._H[i],
                    # H_exp=self._H_exp[i],
                )

            psfs = psfs / torch.norm(psfs.flatten())

        else:

            assert len(slm_vals) == self.n_slm_mask

            # compute masks
            masks_dim = [self.n_slm_mask, self.color_system.n_wavelength] + list(
                self.input_shape[1:]
            )
            masks = torch.zeros(masks_dim, dtype=self.dtype, device=self.device)
            for i in range(self.n_slm_mask):
                masks[i] = get_slm_mask(
                    slm_vals=slm_vals[i].to(self.device_mask_creation),
                    slm_config=self.slm_config,
                    sensor_config=self.sensor_config,
                    crop_fact=self.crop_fact,
                    target_dim=self.input_shape[1:],
                    deadspace=self.deadspace,
                    device=self.device_mask_creation,
                    dtype=self.dtype,
                    requires_grad=self.requires_grad,
                )

            # apply mask
            u_in = masks * self.spherical_wavefront

            # mask to sensor
            if self._H is None:
                # precompute at very start, not dependent on input pattern, just its shape
                # TODO : benchmark how much it actually saves
                self._H = torch.zeros(
                    [self.color_system.n_wavelength] + list(self.input_shape[1:] * 2),
                    dtype=self.ctype,
                    device=self.device,
                )
                for i in range(self.color_system.n_wavelength):
                    self._H[i] = angular_spectrum(
                        u_in=u_in[0][i],
                        wv=self.color_system.wv[i],
                        d1=self.d1,
                        dz=self.mask2sensor,
                        dtype=self.dtype,
                        device=self.device,
                        return_H=True,
                    )

            psfs = torch.zeros(u_in.shape, dtype=self.ctype, device=self.device)
            for n in range(self.n_slm_mask):
                for i in range(self.color_system.n_wavelength):
                    psfs[n][i], _, _ = angular_spectrum(
                        u_in=u_in[n][i],
                        wv=self.color_system.wv[i],
                        d1=self.d1,
                        dz=self.mask2sensor,
                        dtype=self.dtype,
                        device=self.device,
                        H=self._H[i],
                    )

            norm_fact = (
                torch.norm(psfs.flatten(1, 3), dim=1, keepdim=True).unsqueeze(2).unsqueeze(2)
            )
            psfs = psfs / norm_fact

        # intensity psf
        if self.grayscale:
            self._psf = rgb_to_grayscale(torch.square(torch.abs(psfs)))
        else:
            self._psf = torch.square(torch.abs(psfs))

    def name(self):
        if self.n_slm_mask == 1:
            _name = "SLM"
        else:
            _name = f"SLM{self.n_slm_mask}"

        if self.cnn_lite:
            return f"{_name}_CNNLite_{self.cnn_lite}_FCNN{self.hidden}"
        elif self.vgg:
            return f"{_name}_{self.vgg}"
        elif self.n_kern:
            return f"{_name}_CNN_{self.n_kern}"
        elif self.hidden:
            return f"{_name}_SingleHidden{self.hidden}"
        elif self.hidden2:
            return f"{_name}_fully_connected_{self.hidden}_{self.hidden2}"
        else:
            return f"{_name}_MultiClassLogistic"
