# VGG11 used by all models, RGB, SGD 0.01

# Table H.5, first row
TRAIN_HYBRID=false
SCHED=10
PAD_VALS=(4 4 2 0)   # same length as DOWN_ORIG_VALS
PSF_FP=(
    'psfs/lens.png'
)

# # Table H.5, second row
# TRAIN_HYBRID=false
# SCHED=10
# PAD_VALS=(1 1 1 0)   # same length as DOWN_ORIG_VALS
# PSF_FP=(
#     'psfs/simulated_adafruit_deadspaceTrue_15052022_21h04.png' 
#     'psfs/tape.png' 
#     'psfs/adafruit.png' 
#     'psfs/simulated_mls63_mask2sensor0p0005_17052022_18h00_12bit.png'
# )

# # Table H.5, third row
# TRAIN_HYBRID=true
# SCHED=20
# PAD_VALS=(2 1 1 0)   # same length as DOWN_ORIG_VALS
# PSF_FP=()


# ---------------


N_FILES=0     # set to 0 to run all files
N_EPOCH=50
BATCH_SIZE=32
OBJECT_HEIGHT=0.25
SCENE_TO_MASK=0.40
DOWN_PSF_DEFAULT=8
SEED=0
DOWN_ORIG_VALS=(1 4 16 64)
DATA_DIR="data"


len_down_orig=${#DOWN_ORIG_VALS[@]}


for (( i=0; i<$len_down_orig; i++ ));
do
    for psf in "${PSF_FP[@]}"
    do
        down_psf=$DOWN_PSF_DEFAULT
        crop_psf=0
        batch=$BATCH_SIZE

        if [ $psf == 'psfs/simulated_mls63_mask2sensor0p0005_17052022_18h00_12bit.png' ]
        then
            mask2sensor=0.0005
            
        elif [ $psf == 'psfs/lens.png' ]
        then
            mask2sensor=0.00753
            down_psf=1
            crop_psf=100
        else
            mask2sensor=0.004
        fi

        printf "\n-------"
        echo $psf, "down_orig : "${DOWN_ORIG_VALS[$i]}, "mask2sensor : "$mask2sensor, "down_psf : "$down_psf, "crop_psf : "$crop_psf, "batch : "$BATCH_SIZE


        python scripts/train_fixed_encoder.py --down_orig ${DOWN_ORIG_VALS[$i]} ${OUTPUT_DIM_VALS_2[$i]} --psf $psf \
        --batch_size $BATCH_SIZE --n_epoch $N_EPOCH --opti sgd --lr 0.01 --object_height $OBJECT_HEIGHT \
        --mask2sensor $mask2sensor --noise_type poisson --seed $SEED  --aug_pad ${PAD_VALS[$i]} \
        --down_psf $down_psf --device cuda:0 --crop_psf $crop_psf --n_files $N_FILES --vgg VGG11 --rgb \
        --output_dir $DATA_DIR --use_max_range --sched $SCHED --task cifar10 --scene2mask $SCENE_TO_MASK

    done

    if [ "$TRAIN_HYBRID" = true ] ; then
        printf "\n-------"
        down_psf=$DOWN_PSF_DEFAULT
        crop_psf=0
        mask2sensor=0.004
        echo "hybrid", "down_orig : "${DOWN_ORIG_VALS[$i]}, "mask2sensor : "$mask2sensor, "down_psf : "$down_psf, "crop_psf : "$crop_psf, "batch : "$BATCH_SIZE

        python scripts/train_hybrid.py --down_orig ${DOWN_ORIG_VALS[$i]} --sensor_act relu --crop_fact 0.8 \
        --batch_size $BATCH_SIZE --n_epoch $N_EPOCH --opti sgd --lr 0.01 --object_height $OBJECT_HEIGHT --vgg VGG11 \
        --mask2sensor $mask2sensor --noise_type poisson --seed $SEED --aug_pad ${PAD_VALS[$i]} \
        --down_psf $down_psf --device cuda:0 --n_files $N_FILES --output_dir $DATA_DIR --use_max_range \
        --sched $SCHED --task cifar10 --scene2mask $SCENE_TO_MASK --rgb
    fi

done
