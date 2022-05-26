# LenslessClassification

## Setup
```
conda create --name lensless_class python=3.9
pip install -e .
conda install -c pytorch torchvision==0.10.1 cudatoolkit=11.1 pytorch
```

Install `waveprop` from source.

## PSFs

All fixed PSFs can be found in `psfs`.

The simulated PSFs (Coded Aperture and Fixed SLM (s)) are already in this folder. New ones can be simulated as specified below:
- Coded Aperture: `notebooks/mls_mask.ipynb`
- Fixed SLM (s): `save_simulated_psf.py`

## Running experiments

In the following bash scripts, the variable `N_FILES` can be used to run approaches on a small set of files. Set it to `0` to run on all files.

The following script can be used to run the experiments of Section 4.1:
```
./mnist_vary_dimension.sh
```

The following script can be used to run the experiments of Section 4.2:
```
./mnist_robustness.sh
```

Both bash scripts make use of the two training scripts:
-  `scripts/train_fixed_encoder.py`: training a fixed encoder (Lens, Coded Aperture, Diffuser, Fixed SLM (m), Fixed SLM (s)).
-  `scripts/train_hybrid.py`: jointly training SLM with the classifier.
