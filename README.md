# LenslessClassification

Setup
```
conda create --name lensless_class python=3.9
pip install -e .
conda install -c pytorch torchvision==0.10.1 cudatoolkit=11.1 pytorch
```

Install `waveprop` from suource.

## Creating simulated PSFs

- MLS: `notebooks/mls_mask.ipynb`
- Fixed SLM (s): `save_simulated_psf.py`

## Running experiments

The following script can be used to run the experiments of Section 4.1:
```
./mnist_vary_dimension.sh
```

The following script can be used to run the experiments of Section 4.2:
```
./mnist_robustness.sh
```
