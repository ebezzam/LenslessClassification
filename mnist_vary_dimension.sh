PSF_FP=(
    'psfs/simulated_adafruit_deadspaceTrue_15052022_21h04.png' 
    'psfs/tape.png' 
    'psfs/adafruit.png' 
    'psfs/simulated_mls63_mask2sensor0p0005_17052022_18h00_12bit.png'
    'psfs/lens.png'
)

# Table E.2, first row
BATCH_SIZE=64
HIDDEN_VALS=(0)
SCHED=0
TRAIN_HYBRID=false

# # Table E.2, second row
# BATCH_SIZE=100
# HIDDEN_VALS=(0)
# SCHED=0
# TRAIN_HYBRID=true
# PSF_FP=()

# # Table E.2, third row
# BATCH_SIZE=32
# HIDDEN_VALS=(800)
# SCHED=20
# TRAIN_HYBRID=false

# # Table E.2, fourth row
# BATCH_SIZE=64
# HIDDEN_VALS=(800)
# SCHED=20
# TRAIN_HYBRID=true
# PSF_FP=()


# --------


TRAIN_HYBRID=false
N_FILES=0     # set to 0 to run all files
USE_MAX_RANGE=true

N_EPOCH=50
OBJECT_HEIGHT=0.12
DOWN_PSF_DEFAULT=8
SEED=0
DOWN_ORIG_VALS=(1 4 16 64)
DATA_DIR="data"

if (( $N_FILES > 0 ))
then
    # TESTING WITH A FEW FILES
    BATCH_SIZE=$N_FILES
fi

for hidden in "${HIDDEN_VALS[@]}"
do
    for down_orig in "${DOWN_ORIG_VALS[@]}"
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
            echo $psf, "down_orig : "$down_orig, "mask2sensor : "$mask2sensor, "down_psf : "$down_psf, "crop_psf : "$crop_psf, "hidden : "$hidden, "batch : "$BATCH_SIZE

            if [ "$USE_MAX_RANGE" = true ] ; then
                python scripts/train_fixed_encoder.py --psf $psf --down_orig "$down_orig" \
                --batch_size $BATCH_SIZE --n_epoch $N_EPOCH --opti adam --object_height $OBJECT_HEIGHT \
                --mask2sensor $mask2sensor --noise_type poisson --seed $SEED --hidden $hidden \
                --down_psf $down_psf --device cuda:0 --crop_psf $crop_psf --n_files $N_FILES \
                --output_dir $DATA_DIR --use_max_range --sched $SCHED
            else
                python scripts/train_fixed_encoder.py --psf $psf --down_orig "$down_orig" \
                --batch_size $BATCH_SIZE --n_epoch $N_EPOCH --opti adam --object_height $OBJECT_HEIGHT \
                --mask2sensor $mask2sensor --noise_type poisson --seed $SEED --hidden $hidden \
                --down_psf $down_psf --device cuda:0 --crop_psf $crop_psf --n_files $N_FILES \
                --output_dir $DATA_DIR --sched $SCHED
            fi

        done

        if [ "$TRAIN_HYBRID" = true ] ; then
            printf "\n-------"
            down_psf=$DOWN_PSF_DEFAULT
            crop_psf=0
            mask2sensor=0.004
            echo "hybrid", "down_orig : "$down_orig, "mask2sensor : "$mask2sensor, "down_psf : "$down_psf, "crop_psf : "$crop_psf, "hidden : "$hidden, "batch : "$BATCH_SIZE

            if [ "$USE_MAX_RANGE" = true ] ; then
                python scripts/train_hybrid.py --down_orig "$down_orig" --sensor_act relu --crop_fact 0.8 \
                --batch_size $BATCH_SIZE --n_epoch $N_EPOCH --opti adam --object_height $OBJECT_HEIGHT \
                --mask2sensor $mask2sensor --noise_type poisson --seed $SEED --hidden $hidden \
                --down_psf $down_psf --device cuda:0 --n_files $N_FILES --output_dir $DATA_DIR --use_max_range \
                --sched $SCHED
            else
                python scripts/train_hybrid.py --down_orig "$down_orig" --sensor_act relu --crop_fact 0.8 \
                --batch_size $BATCH_SIZE --n_epoch $N_EPOCH --opti adam --object_height $OBJECT_HEIGHT \
                --mask2sensor $mask2sensor --noise_type poisson --seed $SEED --hidden $hidden \
                --down_psf $down_psf --device cuda:0 --n_files $N_FILES --output_dir $DATA_DIR \
                --sched $SCHED
            fi
        fi

    done

done