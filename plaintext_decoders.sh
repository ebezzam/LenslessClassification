# 1 learned mask

python scripts/train_celeba_decoder.py \
--dataset celeba_1_learned_mixed_mask_out768_offset100000_nfiles100000 \
--batch_size 4 --n_epoch 50 --seed 0 \
--device cuda:0 --attr Male --model conv \
--loss l1 --opti sgd --n_files 100

python scripts/train_celeba_decoder.py \
--dataset celeba_1_learned_mixed_mask_out768_offset100000_nfiles100000 \
--batch_size 16 --n_epoch 50 --seed 0 \
--device cuda:0 --attr Male --model conv \
--loss l1 --opti sgd --n_files 1000

python scripts/train_celeba_decoder.py \
--dataset celeba_1_learned_mixed_mask_out768_offset100000_nfiles100000 \
--batch_size 32 --n_epoch 50 --seed 0 \
--device cuda:0 --attr Male --model conv \
--loss l1 --opti sgd --n_files 10000

python scripts/train_celeba_decoder.py \
--dataset celeba_1_learned_mixed_mask_out768_offset100000_nfiles100000 \
--batch_size 32 --n_epoch 50 --seed 0 \
--device cuda:0 --attr Male --model conv \
--loss l1 --opti sgd --n_files 100000


# 10 learned masks

python scripts/train_celeba_decoder.py \
--dataset celeba_10_learned_mixed_mask_out768_offset100000_nfiles100000 \
--batch_size 4 --n_epoch 50 --seed 0 \
--device cuda:0 --attr Male --model conv \
--loss l1 --opti sgd --n_files 100

python scripts/train_celeba_decoder.py \
--dataset celeba_10_learned_mixed_mask_out768_offset100000_nfiles100000 \
--batch_size 16 --n_epoch 50 --seed 0 \
--device cuda:0 --attr Male --model conv \
--loss l1 --opti sgd --n_files 1000

python scripts/train_celeba_decoder.py \
--dataset celeba_10_learned_mixed_mask_out768_offset100000_nfiles100000 \
--batch_size 32 --n_epoch 50 --seed 0 \
--device cuda:0 --attr Male --model conv \
--loss l1 --opti sgd --n_files 10000

python scripts/train_celeba_decoder.py \
--dataset celeba_10_learned_mixed_mask_out768_offset100000_nfiles100000 \
--batch_size 32 --n_epoch 50 --seed 0 \
--device cuda:0 --attr Male --model conv \
--loss l1 --opti sgd --n_files 100000


# 100 random masks

python scripts/train_celeba_decoder.py \
--dataset celeba_100_random_mixed_mask_nonlinTrue_out768_offset100000_nfiles100000 \
--batch_size 4 --n_epoch 50 --seed 0 \
--device cuda:0 --attr Male --model conv \
--loss l1 --opti sgd --n_files 100

python scripts/train_celeba_decoder.py \
--dataset celeba_100_random_mixed_mask_nonlinTrue_out768_offset100000_nfiles100000 \
--batch_size 16 --n_epoch 50 --seed 0 \
--device cuda:0 --attr Male --model conv \
--loss l1 --opti sgd --n_files 1000

python scripts/train_celeba_decoder.py \
--dataset celeba_100_random_mixed_mask_nonlinTrue_out768_offset100000_nfiles100000 \
--batch_size 32 --n_epoch 50 --seed 0 \
--device cuda:0 --attr Male --model conv \
--loss l1 --opti sgd --n_files 10000

python scripts/train_celeba_decoder.py \
--dataset celeba_100_random_mixed_mask_nonlinTrue_out768_offset100000_nfiles100000 \
--batch_size 32 --n_epoch 50 --seed 0 \
--device cuda:0 --attr Male --model conv \
--loss l1 --opti sgd --n_files 100000


# 10 random

python scripts/train_celeba_decoder.py \
--dataset celeba_10_random_mixed_mask_nonlinTrue_out768_offset100000_nfiles100000 \
--batch_size 4 --n_epoch 50 --seed 0 \
--device cuda:0 --attr Male --model conv \
--loss l1 --opti sgd --n_files 100

python scripts/train_celeba_decoder.py \
--dataset celeba_10_random_mixed_mask_nonlinTrue_out768_offset100000_nfiles100000 \
--batch_size 16 --n_epoch 50 --seed 0 \
--device cuda:0 --attr Male --model conv \
--loss l1 --opti sgd --n_files 1000

python scripts/train_celeba_decoder.py \
--dataset celeba_10_random_mixed_mask_nonlinTrue_out768_offset100000_nfiles100000 \
--batch_size 32 --n_epoch 50 --seed 0 \
--device cuda:0 --attr Male --model conv \
--loss l1 --opti sgd --n_files 10000

python scripts/train_celeba_decoder.py \
--dataset celeba_10_random_mixed_mask_nonlinTrue_out768_offset100000_nfiles100000 \
--batch_size 32 --n_epoch 50 --seed 0 \
--device cuda:0 --attr Male --model conv \
--loss l1 --opti sgd --n_files 100000
