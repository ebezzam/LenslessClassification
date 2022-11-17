# LenslessClassification

Code for the paper "Privacy-Enhancing Optical Embeddings for Lensless Classification".

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

## PSFs

All fixed PSFs can be found in `psfs`.

The simulated PSFs (Coded Aperture and Fixed SLM (s)) are already in this folder. New ones can be simulated as specified below:
- Coded Aperture: `notebooks/mls_mask.ipynb`
- Fixed SLM (s): `save_simulated_psf.py`


## End-to-end training

In the following bash scripts, the variable `N_FILES` can be used to run approaches on a small set of files. Set it to `0` to run on all files. *Note that different section need to commented/uncommented as we use different hyperparameters depending on the camera.*

The following script can be used to run the experiments of Section 5.1 (varying embedding dimension). 
```
./mnist_vary_dimension.sh
```

The following script can be used to run the experiments of Section 5.1 (robustness to common image transformations):
```
./mnist_robustness.sh
```

The following script can be used to run the experiments of Section 5.2 (Face attribute classification with CelebA):
```
./celeba_gender_vary_dimension.sh
./celeba_smiling_vary_dimension.sh
```

The following script can be used to run the experiments of Section 5.3 (RGB object classification with CIFAR10):
```
./cifar10_vary_dimension.sh
```

All bash scripts make use of the two training scripts:
-  `scripts/train_fixed_encoder.py`: training a fixed encoder (Lens, Coded Aperture, Diffuser, Fixed SLM (m), Fixed SLM (s)).
-  `scripts/train_hybrid.py`: jointly training SLM with the classifier.

These Python scripts can also be called with user-defined parameters.

## Simulating example embedddings

To simulate and save as PNF embeddings of the different cameras, the following script can be used:
```
python scripts/simulate_examples.py --task mnist --n_files 10
```

Task can be set to `mnist`, `celeba`, or `cifar10`.

Note that some parameters have to be manually set in the script:
- Dimension of simualated examples.
- Which cameras to simulate, by specifiying PSF paths for fixed masks and model maths for learned masks.

The same script is used to plot the PSFs of learned masks.

With the `--cam [KEY]` option, a specific camera can be picked.

With the `--recover [N_ITER]` option, a convex optimization approach can be used to recover the underlying image (only for MNIST and CelebA as RGB isn't supported). Not that only MNIST is discernable for (24x32) as the downsampling is too harsh and the content is too complex for other data. With 100 iterations, the digit can be discerned; 500 iterations gives much better quality.

Perturbations can be applied to MNIST as in Section 5.1 (robustness to common image transformations): `--random_shift`, `--random_height`, `--random_rotate`, `--perspective`.

For example, recover diffuser embeddings with heights between 15 and 20 cm:
```
python scripts/simulate_examples.py --task mnist --recover 200 --cam diffuser --random_height 15 20
```


## Defense to attacks experiments

For convex optimization-based attack, the following command will apply the attack to a set of files (defined by command line):
```
# with knowledge of the mask
python scripts/convex_optimization_attack.py \
--output_dim 384 512 \
--n_files 50

# with random mask
python scripts/convex_optimization_attack.py \
--output_dim 384 512 \
--n_files 50 --diff_slm
```

For the generator-based attack, first a dataset from the multiple masks needs to be generated. For example, to generate a dataset from 10 learned PSFs:
```
python scripts/create_mix_mask_dataset.py --n_mask 10 --learned
```
Note that the paths to the end-to-end models needs to be specified in the script.

Then to train a generator, the dataset should be specified with other training hyperparameters:
```
python scripts/train_celeba_decoder.py \
--dataset data/celeba_10_learned_mixed_mask_out768_offset100000_nfiles100000 \
--batch_size 32 --n_epoch 50 --seed 0 --model conv \
--loss l1 --opti sgd --n_files 100000
```

To generate and save decoded images:
```
python scripts/generate_celeba_examples.py
```
Note that trained generators are specified in the script.


## Notebooks to visualize results

In the `notebooks` folder:

- `1_mnist.ipynb`: compare performance of different cameras, architectures, dimensions on handwritten digit classification (MNIST).
- `2_celeba.ipynb`: compare performance of different cameras and dimensions on face attribute classification (CelebA).
- `3_cifar10.ipynb`: compare performance of different cameras and dimensions on face attribute classification (CelebA).
- `4_convex_optimization_attack.ipynb`: visualize examples of convex optimization-based attack.
- `5_plaintext_generator.ipynb`: compare performance when varying number of plaintext attacks and number of varying masks.

Additional notebooks:
- `simulate_coded_aperture_psf.ipynb` generating coded aperture mask and PSF as in FlatCam paper.


- making MLS mask -> convert as script
- exploring MNIST / celeba datasets?


