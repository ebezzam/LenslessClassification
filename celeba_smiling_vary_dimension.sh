ATTR="Smiling"

# Table G.4, first row
BATCH_SIZE=64
SCHED=10
TRAIN_HYBRID=false
PSF_FP=(
    'psfs/simulated_adafruit_deadspaceTrue_15052022_21h04.png' 
    'psfs/tape.png' 
    'psfs/adafruit.png' 
    'psfs/simulated_mls63_mask2sensor0p0005_17052022_18h00_12bit.png'
    'psfs/lens.png'
)

# # Table G.4, second row
# # -- comment all paths inside `PSF_FP``
# BATCH_SIZE=64
# SCHED=20
# TRAIN_HYBRID=true
# PSF_FP=()


# ------- 


N_FILES=100000     # set to 0 to run all files
N_EPOCH=50
HIDDEN_VALS=(800)
OBJECT_HEIGHT=0.27
SCENE_TO_MASK=0.55
DOWN_PSF_DEFAULT=8
SEED=0
OUTPUT_DIM_VALS_1=(24 3)   # OUTPUT_DIM_VALS_2 to define second dimension
OUTPUT_DIM_VALS_2=(32 4)
DATA_DIR="data"
DATA_DIR="/scratch/LenslessClassification_data/celeba"


len_output_dim_vals=${#OUTPUT_DIM_VALS_1[@]}

for hidden in "${HIDDEN_VALS[@]}"
do
    # for output_dim in "${OUTPUT_DIM_VALS[@]}"
    for (( i=0; i<$len_output_dim_vals; i++ ));
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
            echo $psf, "output_dim : "${OUTPUT_DIM_VALS_1[$i]} ${OUTPUT_DIM_VALS_2[$i]}, "mask2sensor : "$mask2sensor, "down_psf : "$down_psf, "crop_psf : "$crop_psf, "hidden : "$hidden, "batch : "$BATCH_SIZE


            python scripts/train_fixed_encoder.py --output_dim ${OUTPUT_DIM_VALS_1[$i]} ${OUTPUT_DIM_VALS_2[$i]} --psf $psf \
            --batch_size $BATCH_SIZE --n_epoch $N_EPOCH --opti adam --object_height $OBJECT_HEIGHT \
            --mask2sensor $mask2sensor --noise_type poisson --seed $SEED --hidden $hidden  \
            --down_psf $down_psf --device cuda:0 --crop_psf $crop_psf --n_files $N_FILES \
            --output_dir $DATA_DIR --use_max_range --sched $SCHED --task celeba --attr $ATTR --scene2mask $SCENE_TO_MASK

        done

        if [ "$TRAIN_HYBRID" = true ] ; then
            printf "\n-------"
            down_psf=$DOWN_PSF_DEFAULT
            crop_psf=0
            mask2sensor=0.004
            echo "hybrid", "output_dim : "${OUTPUT_DIM_VALS_1[$i]} ${OUTPUT_DIM_VALS_2[$i]}, "mask2sensor : "$mask2sensor, "down_psf : "$down_psf, "crop_psf : "$crop_psf, "hidden : "$hidden, "batch : "$BATCH_SIZE

            python scripts/train_hybrid.py --output_dim ${OUTPUT_DIM_VALS_1[$i]} ${OUTPUT_DIM_VALS_2[$i]} --sensor_act relu --crop_fact 0.8 \
            --batch_size $BATCH_SIZE --n_epoch $N_EPOCH --opti adam --object_height $OBJECT_HEIGHT \
            --mask2sensor $mask2sensor --noise_type poisson --seed $SEED --hidden $hidden \
            --down_psf $down_psf --device cuda:0 --n_files $N_FILES --output_dir $DATA_DIR --use_max_range \
            --sched $SCHED --task celeba --attr $ATTR --scene2mask $SCENE_TO_MASK
        fi

    done

done