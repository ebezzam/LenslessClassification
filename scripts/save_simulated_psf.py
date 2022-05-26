"""
Simulate PSF at sensor full resolution
"""

from lenslessclass.models import SLMMultiClassLogistic
from waveprop.devices import slm_dict, sensor_dict, SensorParam
import cv2
import numpy as np
from lensless.util import print_image_info
import time
from datetime import datetime


down_out = 1
sensor = "rpi_hq"
slm = "adafruit"
crop_fact = 0.8
device = "cpu"
device_mask_creation = "cpu"
deadspace = False
scene2mask = 0.4
mask2sensor = 0.004
bit_depth = 12

timestamp = datetime.now().strftime("%d%m%Y_%Hh%M")

sensor_param = sensor_dict[sensor]
sensor_size = sensor_param[SensorParam.SHAPE]
if down_out > 1:
    sensor_size = (sensor_size * 1 / down_out).astype(int)

start_time = time.time()
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
    n_class=10,
    grayscale=False,
    requires_grad=False,
)
print("Computation time [m] : ", (time.time() - start_time) / 60.0)

psf_sim = model._psf.numpy()
psf_sim = np.transpose(psf_sim, (1, 2, 0))
print("SLM dimensions : ", model.slm_vals.shape)
print("Number of SLM pixels : ", np.prod(model.slm_vals.shape))

# cast to uint as on sensor
psf_sim /= psf_sim.max()
psf_sim *= 2**bit_depth - 1
psf_sim = psf_sim.astype(dtype=np.uint16)
print_image_info(psf_sim)

fp = f"simulated_adafruit_deadspace{deadspace}_{timestamp}.png"
cv2.imwrite(fp, cv2.cvtColor(psf_sim, cv2.COLOR_RGB2BGR))
print("Saved simulated PSF to : ", fp)
