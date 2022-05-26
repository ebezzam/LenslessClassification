from audioop import bias
from os import device_encoding
from torch import nn
import torch
import warnings
from torchvision import transforms
from torchvision.transforms.functional import rgb_to_grayscale
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


class MultiClassLogistic(nn.Module):
    """
    Example: https://gist.github.com/xmfbit/b27cdbff68870418bdb8cefa86a2d558
    """

    def __init__(self, input_shape, multi_gpu=False):
        super(MultiClassLogistic, self).__init__()
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(int(np.prod(input_shape)), 10)
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

    def __init__(self, input_shape, hidden_dim, n_class):
        assert isinstance(hidden_dim, int)
        self.hidden_dim = hidden_dim
        super(SingleHidden, self).__init__()
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(int(np.prod(input_shape)), hidden_dim, bias=False)
        self.bn1 = nn.BatchNorm1d(hidden_dim)  # ADDED to be consistent with SLM
        self.activation1 = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, n_class)
        if n_class > 1:
            self.decision = nn.Softmax(dim=1)
        else:
            self.decision = nn.Sigmoid()

    def forward(self, x):
        x = self.flatten(x)
        # x = self.activation1(self.linear1(x))
        x = self.activation1(self.bn1(self.linear1(x)))  # ADDED to be consistent with SLM
        x = self.linear2(x)
        logits = self.decision(x)
        return logits

    def name(self):
        return f"SingleHidden{self.hidden_dim}"


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

        # -- normalize after PSF
        self.conv_bn = nn.BatchNorm2d(1)  # 3 if RGB

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

            if down == "resize":
                self.downsample = transforms.Resize(size=self.output_dim.tolist())

            elif down == "max" or down == "avg":
                n_embedding = np.prod(self.output_dim)

                # determine filter size, stride, and padding: https://androidkt.com/calculate-output-size-convolutional-pooling-layers-cnn/
                k = int(np.ceil(np.sqrt(np.prod(self.input_shape) / n_embedding)))
                p = np.roots(
                    [
                        4,
                        2 * np.sum(self.input_shape),
                        np.prod(self.input_shape) - k**2 * n_embedding,
                    ]
                )
                p = max(int(np.max(p)), 0) + 1
                if down == "max":
                    self.downsample = nn.MaxPool2d(kernel_size=k, stride=k, padding=p)
                else:
                    self.downsample = nn.AvgPool2d(kernel_size=k, stride=k, padding=p)
                pooling_outdim = ((self.input_shape - k + 2 * p) / k + 1).astype(int)
                assert np.array_equal(self.output_dim, pooling_outdim)
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

        # -- remove bias from linear layers before batch norm: https://stackoverflow.com/a/59736545
        # linear_layers = []
        # if self.hidden is not None:
        #     linear_layers += [
        #         ('linear1', nn.Linear(self.n_hidden, self.hidden, bias=False)),
        #         ('bn', nn.BatchNorm1d(self.hidden)),
        #         ('relu1', nn.ReLU()),
        #     ]
        #     if self.hidden2 is not None:
        #         linear_layers += [
        #             ('linear2', nn.Linear(self.hidden, self.hidden2, bias=False)),
        #             ('bn', nn.BatchNorm1d(self.hidden2)),
        #             ('relu1', nn.ReLU()),
        #             ('linear3', nn.Linear(self.hidden2, n_class)),
        #         ]
        #     else:
        #         linear_layers.append(
        #             ('linear2', nn.Linear(self.hidden, n_class))
        #         )
        # else:
        #     linear_layers.append(
        #         ('linear1', nn.Linear(self.n_hidden, self.hidden, bias=False))
        #     )
        # self.linear_layers = nn.Sequential(OrderedDict(linear_layers))

        if self.hidden2 is not None:
            assert self.hidden is not None
            self.linear_layers = nn.Sequential(
                OrderedDict(
                    [
                        ("linear1", nn.Linear(self.n_hidden, self.hidden, bias=False)),
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
            if self.hidden is not None:
                self.linear1 = nn.Linear(self.n_hidden, self.hidden, bias=False)
                self.bn = nn.BatchNorm1d(self.hidden)
                self.linear2 = nn.Linear(self.hidden, n_class)
            else:
                self.linear1 = nn.Linear(self.n_hidden, n_class)
                self.bn = None
                self.linear2 = None
            self.linear_layers = None

        if n_class > 1:
            self.decision = nn.Softmax(dim=1)
        else:
            self.decision = nn.Sigmoid()

        if multi_gpu:
            self.downsample = nn.DataParallel(self.downsample, device_ids=multi_gpu)
            if self.linear_layers is not None:
                self.linear_layers = nn.DataParallel(self.linear_layers, device_ids=multi_gpu)
            else:
                self.conv_bn = nn.DataParallel(self.conv_bn, device_ids=multi_gpu)
                self.linear1 = nn.DataParallel(self.linear1, device_ids=multi_gpu)
                if self.linear2:
                    self.linear2 = nn.DataParallel(self.linear2, device_ids=multi_gpu)
                    self.bn = nn.DataParallel(self.bn, device_ids=multi_gpu)

        # -- determine number of active SLM pixels
        overlapping_mask_size, overlapping_mask_dim, n_active_slm_pixels = get_active_pixel_dim(
            sensor_config=sensor_config,
            sensor_crop=crop_fact,
            slm_config=slm_config,
        )

        # -- initialize SLM values, set as parameter to optimize
        # TODO : need to make internal changes to get_slm_mask for no deadspace
        if deadspace:
            if self.requires_grad:
                self.slm_vals = nn.Parameter(
                    torch.rand(
                        *n_active_slm_pixels,
                        dtype=dtype,
                        device=self.device,
                        requires_grad=self.requires_grad,
                    )
                )
            else:
                self.slm_vals = torch.rand(
                    *n_active_slm_pixels,
                    dtype=dtype,
                    device=self.device,
                    requires_grad=self.requires_grad,
                )
        else:
            if self.requires_grad:
                self.slm_vals = nn.Parameter(
                    torch.rand(
                        *overlapping_mask_dim,
                        dtype=dtype,
                        device=self.device,
                        requires_grad=self.requires_grad,
                    )
                )
            else:
                self.slm_vals = torch.rand(
                    *overlapping_mask_dim,
                    dtype=dtype,
                    device=self.device,
                    requires_grad=self.requires_grad,
                )

        # -- initialize PSF from SLM values and pre-compute constants
        # object to mask (modeled with spherical propagation which can be pre-computed)
        self.color_system = ColorSystem.rgb()
        self.d1 = np.array(overlapping_mask_size) / input_shape
        self.spherical_wavefront = spherical_prop(
            in_shape=input_shape,
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
        self.compute_intensity_psf()

    @property
    def psf(self, numpy=False):
        if numpy:
            self._psf.cpu().detach().numpy().squeeze()
        else:
            return self._psf

    def forward(self, x):

        if x.min() < 0:
            warnings.warn("Got negative data. Shift to non-negative.")
            x -= x.min()

        # TODO : compute PSF here?? on in training loop to give user
        #  flexibility of how often to update SLM mask
        # - convolve with intensity PSF
        x = fftconvolve(x, self._psf, axes=(-2, -1))

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

        # -- digital decision network after sensor
        x = self.flatten(x)
        if self.linear_layers is not None:
            x = self.linear_layers(x)
        else:
            x = self.linear1(x)
            if self.dropout is not None:
                x = self.dropout(x)
            if self.hidden:
                x = self.sensor_activation(self.bn(x))
                x = self.linear2(x)
        logits = self.decision(x)
        return logits

    def compute_intensity_psf(self):
        """
        TODO precompute things when first called!! spherical prop, H matrix, etc
        :return:
        """

        # -- get SLM mask, i.e. deadspace modeling, quantization (todo), non-linearities (todo), etc
        mask = get_slm_mask(
            slm_config=self.slm_config,
            sensor_config=self.sensor_config,
            crop_fact=self.crop_fact,
            target_dim=self.input_shape,
            slm_vals=self.slm_vals.to(self.device_mask_creation),
            deadspace=self.deadspace,
            device=self.device_mask_creation,
            dtype=self.dtype,
            first_color=self.first_color,
            requires_grad=self.requires_grad,
        )

        # TODO can variable be on different device
        mask = mask.to(self.device)

        # apply mask
        u_in = mask * self.spherical_wavefront

        # mask to sensor
        # if self._H_exp is None:
        #     # pre-compute H_exp if trying to optimize distance, TODO check if mask2sensor is a tensor
        #     self._H_exp = torch.zeros(
        #         [3] + list(self.input_shape * 2), dtype=self.ctype, device=self.device
        #     )
        #     for i in range(self.color_system.n_wavelength):
        #         self._H_exp[i] = angular_spectrum(
        #             u_in=u_in[i],
        #             wv=self.color_system.wv[i],
        #             d1=self.d1,
        #             dz=self.mask2sensor,
        #             dtype=self.dtype,
        #             device=self.device,
        #             return_H_exp=True,
        #         )
        if self._H is None:
            # TODO : benchmark how much it actually saves
            self._H = torch.zeros(
                [3] + list(self.input_shape * 2), dtype=self.ctype, device=self.device
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

        # intensity psf
        if self.grayscale:
            self._psf = rgb_to_grayscale(torch.square(torch.abs(psfs)))
        else:
            self._psf = torch.square(torch.abs(psfs))

    def name(self):
        if self.hidden:
            return f"SLMSingleHidden{self.hidden}"
        elif self.hidden2:
            return f"SLM_fully_connected_{self.hidden}_{self.hidden2}"
        else:
            return "SLMMultiClassLogistic"
