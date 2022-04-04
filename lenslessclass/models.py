from torch import nn
import torch
from torchvision.transforms.functional import rgb_to_grayscale
import numpy as np
from waveprop.slm import get_active_pixel_dim, get_slm_mask
from waveprop.pytorch_util import fftconvolve
from waveprop.spherical import spherical_prop
from waveprop.color import ColorSystem
from waveprop.rs import angular_spectrum


class MultiClassLogistic(nn.Module):
    """
    Example: https://gist.github.com/xmfbit/b27cdbff68870418bdb8cefa86a2d558
    """

    def __init__(self, input_shape):
        super(MultiClassLogistic, self).__init__()
        self.flatten = nn.Flatten()
        self.multiclass_logistic_reg = nn.Sequential(
            nn.Linear(int(np.prod(input_shape)), 10),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.multiclass_logistic_reg(x)
        return logits

    def name(self):
        return "MultiClassLogistic"


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
    ):
        super(SLMMultiClassLogistic, self).__init__()

        if dtype == torch.float32:
            self.ctype = torch.complex64
        elif dtype == torch.float64:
            self.ctype = torch.complex128
        else:
            raise ValueError(f"Unsupported data type : {dtype}")

        # store configuration
        self.input_shape = input_shape
        self.slm_config = slm_config
        self.sensor_config = sensor_config
        self.crop_fact = crop_fact
        self.scene2mask = scene2mask
        self.mask2sensor = mask2sensor
        self.deadspace = deadspace
        self.first_color = first_color
        self.grayscale = grayscale
        self.device = device
        self.dtype = dtype
        self._numel = int(np.prod(np.array(self.input_shape)))

        # -- decision network after sensor
        self.flatten = nn.Flatten()
        self.multiclass_logistic_reg = nn.Sequential(
            nn.Linear(int(np.prod(input_shape)), 10),
            nn.Softmax(dim=1),
        )

        # normalize after PSF
        self.conv_bn = nn.BatchNorm2d(1)

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
                torch.rand(*n_active_slm_pixels, dtype=dtype, device=device, requires_grad=True)
            )
        else:
            self.slm_vals = nn.Parameter(
                torch.rand(*overlapping_mask_dim, dtype=dtype, device=device, requires_grad=True)
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

    def forward(self, x):

        # TODO : compute PSF here?? on in training loop to give user
        #  flexibility of how often to update SLM mask

        # - convolve with intensity PSF
        x = fftconvolve(x, self._psf, axes=(-2, -1))
        # x = fftconvolve(x, self._psf, axes=(-2, -1)) / self._numel
        x = self.conv_bn(torch.clip(x, min=0))

        # -- digital decision network after sensor
        x = self.flatten(x)
        logits = self.multiclass_logistic_reg(x)
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
            slm_vals=self.slm_vals,
            deadspace=self.deadspace,
            device=self.device,
            dtype=self.dtype,
            first_color=self.first_color,
        )

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

        print(self._psf.max())

    def name(self):
        return "SLMMultiClassLogistic"
