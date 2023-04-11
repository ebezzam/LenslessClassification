# ## First create mixed datasets for offset = 0 
# python scripts/create_mix_mask_dataset.py --n_mask 1 --offset 0 --learned
# # -- data/celeba_1_learned_mixed_mask_out768_offset0_nfiles100000
# python scripts/create_mix_mask_dataset.py --n_mask 10 --offset 0 --learned
# # -- data/celeba_10_learned_mixed_mask_out768_offset0_nfiles100000
# python scripts/create_mix_mask_dataset.py --n_mask 100 --offset 0
# # -- data/celeba_100_random_mixed_mask_nonlinTrue_out768_offset0_nfiles100000

# 100K plaintext attacks
# Male
python scripts/train_plaintext_classifier.py --n_masks 1
python scripts/train_plaintext_classifier.py --n_masks 10
python scripts/train_plaintext_classifier.py --n_masks 100

# Smiling
python scripts/train_plaintext_classifier.py --n_masks 1 --attr Smiling
python scripts/train_plaintext_classifier.py --n_masks 10 --attr Smiling
python scripts/train_plaintext_classifier.py --n_masks 100 --attr Smiling

## 100 plaintext attacks
# Male
python scripts/train_plaintext_classifier.py --n_masks 1 --n_plaintext 100
python scripts/train_plaintext_classifier.py --n_masks 10 --n_plaintext 100
python scripts/train_plaintext_classifier.py --n_masks 100 --n_plaintext 100

# Smiling
python scripts/train_plaintext_classifier.py --n_masks 1 --attr Smiling --n_plaintext 100
python scripts/train_plaintext_classifier.py --n_masks 10 --attr Smiling --n_plaintext 100
python scripts/train_plaintext_classifier.py --n_masks 100 --attr Smiling --n_plaintext 100

## 1K plaintext attacks
# Male
python scripts/train_plaintext_classifier.py --n_masks 1 --n_plaintext 1000
python scripts/train_plaintext_classifier.py --n_masks 10 --n_plaintext 1000
python scripts/train_plaintext_classifier.py --n_masks 100 --n_plaintext 1000

# Smiling
python scripts/train_plaintext_classifier.py --n_masks 1 --attr Smiling --n_plaintext 1000
python scripts/train_plaintext_classifier.py --n_masks 10 --attr Smiling --n_plaintext 1000
python scripts/train_plaintext_classifier.py --n_masks 100 --attr Smiling --n_plaintext 1000

## 10K plaintext attacks
# Male
python scripts/train_plaintext_classifier.py --n_masks 1 --n_plaintext 10000
python scripts/train_plaintext_classifier.py --n_masks 10 --n_plaintext 10000
python scripts/train_plaintext_classifier.py --n_masks 100 --n_plaintext 10000

# Smiling
python scripts/train_plaintext_classifier.py --n_masks 1 --attr Smiling --n_plaintext 10000
python scripts/train_plaintext_classifier.py --n_masks 10 --attr Smiling --n_plaintext 10000
python scripts/train_plaintext_classifier.py --n_masks 100 --attr Smiling --n_plaintext 10000


