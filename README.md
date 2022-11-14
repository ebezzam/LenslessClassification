# LenslessClassification

## Setup
```
conda create --name lensless_class python=3.9
pip install -e .
conda install pytorch torchvision cudatoolkit=11.3 -c pytorch
```

For PyTorch, you can check CUDA version with
```
nvcc --version
```
And find the appropriate installation command [here](https://pytorch.org/).

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


## Defense to attacks experiments

For convex optimization-based attack, the following command will apply the attack to a set of files (defined by command line):
```
# with knowledge of the mask
python scripts/recover_from_downsampled_dataset.py \
--output_dim 384 512 \
--n_files 50

# with random mask
python scripts/recover_from_downsampled_dataset.py \
--output_dim 384 512 \
--n_files 50 --diff_slm
```

For the generator-based attack, first a dataset from the multiple masks needs to be generated. For example, to generate a dataset from 10 learned PSFs:
```
python scripts/create_mix_mask_dataset.py --n_masks 10 --learned
```
Note that the paths to the end-to-end models needs to be specified in the script.

Then to train a generator, the dataset shulud be specified with other training hyperparameters:
```
python scripts/train_celeba_decoder.py \
--dataset celeba_10_learned_mixed_mask_out768_offset100000_nfiles100000 \
--batch_size 32 --n_epoch 50 --seed 0 --model conv \
--loss l1 --opti sgd --n_files 100000
```


