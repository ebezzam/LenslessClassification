# LenslessClassification

Setup
```
conda create --name lensless_class python=3.9
pip install -e .
pip install git+https://github.com/LCAV/LenslessPiCam.git
conda install -c pytorch torchvision==0.10.1 cudatoolkit=11.1 pytorch
```

Have to install `waveprop` as well! Atm a private repo...

Procedure (for fixed PSF):
1. Prepare dataset: `scripts/save_simulated_dataset.py`
2. Run training: `scripts/train_logistic_reg.py`
