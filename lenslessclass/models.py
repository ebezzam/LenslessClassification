from os import device_encoding
from turtle import hideturtle, pu
from torch import nn
import torch
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
            self.linear1 = nn.DataParallel(self.linear1)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear1(x)
        logits = self.decision(x)
        return logits

    def name(self):
        return "MultiClassLogistic"


class SingleHidden(nn.Module):
    """
    Example: https://gist.github.com/xmfbit/b27cdbff68870418bdb8cefa86a2d558
    """

    def __init__(self, input_shape, hidden_dim, multi_gpu=False):
        assert isinstance(hidden_dim, int)
        self.hidden_dim = hidden_dim
        super(MultiClassLogistic, self).__init__()
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(int(np.prod(input_shape)), hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, 10)
        self.decision = nn.Softmax(dim=1)

        if multi_gpu:
            self.linear1 = nn.DataParallel(self.linear1)
            self.linear2 = nn.DataParallel(self.linear2)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.linear2(x)
        logits = self.decision(x)
        return logits

    def name(self):
        return f"SingleHidden{self.hidden_dim}"


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
        device="cpu",
        dtype=torch.float32,
        deadspace=True,
        first_color=0,
        grayscale=True,
        device_mask_creation=None,
        output_dim=None,
        multi_gpu=False,
        sensor_activation=None,
        dropout=None,
        noise_type=None,
        snr=40,
    ):
        super(SLMMultiClassLogistic, self).__init__()

        if dtype == torch.float32:
            self.ctype = torch.complex64
        elif dtype == torch.float64:
            self.ctype = torch.complex128
        else:
            raise ValueError(f"Unsupported data type : {dtype}")

        # store configuration
        self.input_shape = np.array(input_shape)
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
        self.output_dim = np.array(output_dim)
        self._numel = int(np.prod(np.array(self.input_shape)))

        # adding noise
        if noise_type is not None:
            # TODO : hardcoded for Raspberry Pi HQ sensor
            bit_depth = 12
            noise_mean = RPI_HQ_CAMERA_BLACK_LEVEL / (2**bit_depth - 1)

            def add_noise(measurement):

                measurement_copy = measurement.clone().cpu().detach().numpy()
                sig_var = np.linalg.norm(measurement_copy, axis=(-2, -1))
                noise = []
                for i, _val in enumerate(sig_var):

                    noise_var = _val / (10 ** (snr / 10))
                    noise.append(
                        random_noise(
                            measurement_copy[i],
                            mode=noise_type,
                            clip=False,
                            mean=noise_mean,
                            var=noise_var**2,
                        )
                        - measurement_copy[i]
                        # random_noise(
                        #     measurement_copy[i],
                        #     mode=noise_type,
                        #     clip=False,
                        #     mean=noise_mean,
                        #     var=noise_var**2,
                        # )
                    )

                noise = torch.tensor(np.array(noise).astype(np.float32)).to(device)

                # import pudb; pudb.set_trace()

                # # measurement is intensity
                # sig_var = np.linalg.norm(measurement)
                # noise_var = sig_var / (10 ** (snr / 10))
                # noise = torch.tensor(
                #     random_noise(
                #         measurement,
                #         mode=noise_type,
                #         clip=False,
                #         mean=noise_mean,
                #         var=noise_var**2,
                #     ).astype(np.float32)
                # ).to(device)

                return measurement + noise

            self.add_noise = add_noise
        else:
            self.add_noise = None

        # -- normalize after PSF
        self.conv_bn = nn.BatchNorm2d(1)

        # -- decision network after sensor
        if self.output_dim is not None:
            self.downsample = transforms.Resize(size=list(output_dim))

        self.sensor_activation = sensor_activation
        self.flatten = nn.Flatten()
        if dropout is not None:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None
        self.linear1 = nn.Linear(int(np.prod(output_dim)), 10)
        self.decision = nn.Softmax(dim=1)

        if multi_gpu:
            self.downsample = nn.DataParallel(self.downsample)
            self.conv_bn = nn.DataParallel(self.conv_bn)
            self.linear1 = nn.DataParallel(self.linear1)

        # -- determine number of active SLM pixels
        overlapping_mask_size, overlapping_mask_dim, n_active_slm_pixels = get_active_pixel_dim(
            sensor_config=sensor_config,
            sensor_crop=crop_fact,
            slm_config=slm_config,
        )

        # -- initialize SLM values, set as parameter to optimize
        # TODO : need to make internal changes to get_slm_mask for no deadspace
        if deadspace:
            self.slm_vals = nn.Parameter(
                torch.rand(
                    *n_active_slm_pixels, dtype=dtype, device=self.device, requires_grad=True
                )
            )
        else:
            self.slm_vals = nn.Parameter(
                torch.rand(
                    *overlapping_mask_dim, dtype=dtype, device=self.device, requires_grad=True
                )
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

        # TODO : compute PSF here?? on in training loop to give user
        #  flexibility of how often to update SLM mask
        # - convolve with intensity PSF
        x = fftconvolve(x, self._psf, axes=(-2, -1))

        if self.add_noise is not None:
            x = self.add_noise(x)

        # TODO : try other downsampling / pooling schemes
        if self.output_dim is not None:
            x = self.downsample(x)

        x = self.conv_bn(torch.clip(x, min=0))

        if self.sensor_activation is not None:
            x = self.sensor_activation(x)

        if self.dropout is not None:
            x = self.dropout(x)

        # -- digital decision network after sensor
        x = self.flatten(x)
        x = self.linear1(x)
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
        return "SLMMultiClassLogistic"
