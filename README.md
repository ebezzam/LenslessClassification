# Lensless Classification

Code for the paper ["Privacy-Enhancing Optical Embeddings for Lensless Classification"](https://arxiv.org/abs/2211.12864) by Eric Bezzam, Martin Vetterli, and Matthieu Simeoni.

- [Setup](#setup).
- [Point spread functions](#psfs).
- [End-to-end training](#e2e).
- [Simulating example embeddings](#examples).
- [Defense to adversarial attacks](#defense).
- [Notebooks to visualize results](#viz).

## Setup  <a name="setup"></a>
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

There is also an [environment.yml](https://github.com/ebezzam/LenslessClassification/blob/main/environment.yml) file.

We provide our trained models at this [link](https://drive.google.com/file/d/1INX8eYtsNa1z8htH2IgB6tf2yuv2GPqs/view?usp=share_link). Place the `models` directory at the root of this repository so that they can be properly loaded by the various scripts/notebooks. 

### Data

Note we do not provided the augmented datasets as they can be generated when running the training scripts (`scripts/train_fixed_encoder.py` and `scripts/train_hybrid.py`). These scripts will download the original MNIST and CIFAR10 datasets to `data` if they do not already exist locally, and will simulate the corresponding embedding dataset if it does not exist already. 

Note that the original [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) dataset should be downloaded manually.


## PSFs <a name="psfs"></a>

All fixed PSFs can be found in `psfs`.

The simulated PSFs (Coded aperture and Fixed mask (s)) are already in this folder. New ones can be simulated in the following notebook:
- Coded aperture: [`notebook/simulate_coded_aperture_psf.ipynb`](https://github.com/ebezzam/LenslessClassification/blob/main/notebooks/simulate_coded_aperture_psf.ipynb) for generating coded aperture mask and PSF as in the [FlatCam paper](https://arxiv.org/abs/1509.00116).
- Fixed mask (s): [`notebooks/simulate_fixed_mask_psf.ipynb`](https://github.com/ebezzam/LenslessClassification/blob/main/notebooks/simulate_fixed_mask_psf.ipynb) for generating mask and simulated PSF for proposed system.


## End-to-end training <a name="e2e"></a>

*Note that in the following bash scripts, different sections (at the top) need to commented/uncommented as we use different hyperparameters depending on the camera.*

The following script can be used to run the experiments of Section 5.1 (varying embedding dimension). 
```
./mnist_vary_dimension.sh
```

The following script can be used to run the experiments of Section 5.1 (robustness to common image transformations):
```
./mnist_robustness.sh
```

The following scripts can be used to run the experiments of Section 5.2 (Face attribute classification with CelebA):
```
./celeba_gender_vary_dimension.sh
./celeba_smiling_vary_dimension.sh
```

The following script can be used to run the experiments of Section 5.3 (RGB object classification with CIFAR10):
```
./cifar10_vary_dimension.sh
```

All bash scripts make use of the two training scripts:
-  `scripts/train_fixed_encoder.py`: training a fixed encoder (Lens, Coded Aperture, Diffuser, Fixed mask (m), Fixed mask (s)).
-  `scripts/train_hybrid.py`: jointly learning mask with the classifier.

These Python scripts are called with user-defined parameters that need to be set accordingly for each experiment.

The variable `N_FILES` can be used to run approaches on a small set of files. Set it to `0` to run on all files. 

## Simulating example embedddings <a name="examples"></a>

To simulate the embeddings of the different cameras and save them as PNGs, the following script can be used:
```
python scripts/simulate_examples.py --task mnist --n_files 10
```

Task can be set to `mnist`, `celeba`, or `cifar10`.

Note that some parameters have to be manually set in the script:
- Dimension of simualated examples.
- Which cameras to simulate, by specifiying PSF paths for fixed masks and model maths for learned masks.

The same script is used to plot the PSFs of learned masks.

With the `--cam [KEY]` option, a specific camera can be picked.

With the `--recover [N_ITER]` option, a convex optimization approach can be used to recover the underlying image (only for MNIST and CelebA as RGB isn't supported). Not that only MNIST is discernable for (24x32) as the downsampling is too severe and the content is too complex for other data. With 100 iterations, the digit can be discerned; 500 iterations gives much better quality.

Perturbations can be applied to MNIST as in Section 5.1 (robustness to common image transformations): `--random_shift`, `--random_height`, `--random_rotate`, `--perspective`.

For example, recovering diffuser embeddings with heights between 15 and 20 cm:
```
python scripts/simulate_examples.py --task mnist --recover 200 --cam diffuser --random_height 15 20
```


## Defense to adversarial attacks <a name="defense"></a>

These scripts are used to generate the results for Section 5.4 (Defense against leaks and plaintext attacks).

### Convex optimization-based

For convex optimization-based attacks (inverse problem formulation), the following command will apply the attack to a set of CelebA files:
```
# with knowledge of the mask
python scripts/convex_optimization_attack.py \
--output_dim 384 512 \
--n_files 50

# with random mask
python scripts/convex_optimization_attack.py \
--output_dim 384 512 \
--n_files 50 --diff_mask
```

The following bash script can be used to reproduce the results in the paper (Table 4):
```
./convex_opt_attack.sh
```

### Training a decoder

For the generator-based attack, first a dataset from the multiple masks needs to be generated. For example, to generate a dataset from 10 learned PSFs:
```
python scripts/create_mix_mask_dataset.py --n_mask 10 --learned
```
Note that the paths to the end-to-end models needs to be specified in the script.

Then to train a generator, the dataset should be specified along with other training hyperparameters:
```
python scripts/train_celeba_decoder.py \
--dataset data/celeba_10_learned_mixed_mask_out768_offset100000_nfiles100000 \
--batch_size 32 --n_epoch 50 --seed 0 --model conv \
--loss l1 --opti sgd --n_files 100000
```

With all the mixed-mask datasets created, the following bash script can be used to train the plaintext generators for a varying number of masks and plaintext attacks (Table 5):
```
./plaintext_decoders.sh
```

To generate and save decoded images:
```
python scripts/generate_celeba_examples.py
```
Note that trained generators are specified in the script.


## Notebooks to visualize results <a name="viz"></a>

In the `notebooks` folder:

- [`1_mnist.ipynb`](https://github.com/ebezzam/LenslessClassification/blob/main/notebooks/1_mnist.ipynb): compare performance of different cameras, architectures, dimensions on handwritten digit classification (MNIST).
- [`2_celeba.ipynb`](https://github.com/ebezzam/LenslessClassification/blob/main/notebooks/2_celeba.ipynb): compare performance of different cameras and dimensions on face attribute classification (CelebA).
- [`3_cifar10.ipynb`](https://github.com/ebezzam/LenslessClassification/blob/main/notebooks/3_cifar10.ipynb): compare performance of different cameras and dimensions on RGB object classification (CIFAR10).
- [`4_convex_optimization_attack.ipynb`](https://github.com/ebezzam/LenslessClassification/blob/main/notebooks/4_convex_optimization_attack.ipynb): visualize examples of convex optimization-based attack.
- [`5_plaintext_generator.ipynb`](https://github.com/ebezzam/LenslessClassification/blob/main/notebooks/5_plaintext_generator.ipynb): compare decoder performance when varying number of plaintext attacks and number of varying masks.


