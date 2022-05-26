PSF_FP=(
    # 'psfs/tape.png' 
    # 'psfs/adafruit.png' 
    # 'psfs/simulated_adafruit_deadspaceTrue_15052022_21h04.png' 
    # 'psfs/simulated_mls63_mask2sensor0p0005_17052022_18h00_12bit.png'
    # 'psfs/lens.png'
)
TRAIN_HYBRID=true
N_FILES=0     # set to 0 to run all files

N_EPOCH=50
BATCH_SIZE=200
OBJECT_HEIGHT=0.12
DOWN_PSF_DEFAULT=8
SEED=0
DOWN_ORIG_VALS=(1 16)
HIDDEN_VALS=(800)

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

            printf "\n------- SHIFT\n"
            echo $psf, "down_orig : "$down_orig, "mask2sensor : "$mask2sensor, "down_psf : "$down_psf, "crop_psf : "$crop_psf, "hidden : "$hidden, "batch : "$BATCH_SIZE

            python scripts/train_fixed_encoder.py --psf $psf --down_orig $down_orig \
            --batch_size $BATCH_SIZE --n_epoch $N_EPOCH --opti adam --object_height $OBJECT_HEIGHT \
            --mask2sensor $mask2sensor --noise_type poisson --seed $SEED --hidden $hidden \
            --down_psf $down_psf --device cuda:0 --crop_psf $crop_psf --n_files $N_FILES --shift

            printf "\n------- RESCALE\n"
            echo $psf, "down_orig : "$down_orig, "mask2sensor : "$mask2sensor, "down_psf : "$down_psf, "crop_psf : "$crop_psf, "hidden : "$hidden, "batch : "$BATCH_SIZE

            python scripts/train_fixed_encoder.py --psf $psf --down_orig $down_orig \
            --batch_size $BATCH_SIZE --n_epoch $N_EPOCH --opti adam \
            --mask2sensor $mask2sensor --noise_type poisson --seed $SEED --hidden $hidden \
            --down_psf $down_psf --device cuda:0 --crop_psf $crop_psf --n_files $N_FILES --random_height 2 20

            printf "\n------- ROTATE\n"
            echo $psf, "down_orig : "$down_orig, "mask2sensor : "$mask2sensor, "down_psf : "$down_psf, "crop_psf : "$crop_psf, "hidden : "$hidden, "batch : "$BATCH_SIZE

            python scripts/train_fixed_encoder.py --psf $psf --down_orig $down_orig \
            --batch_size $BATCH_SIZE --n_epoch $N_EPOCH --opti adam --object_height $OBJECT_HEIGHT \
            --mask2sensor $mask2sensor --noise_type poisson --seed $SEED --hidden $hidden \
            --down_psf $down_psf --device cuda:0 --crop_psf $crop_psf --n_files $N_FILES --rotate 90

            printf "\n------- PERSPECTIVE\n"
            echo $psf, "down_orig : "$down_orig, "mask2sensor : "$mask2sensor, "down_psf : "$down_psf, "crop_psf : "$crop_psf, "hidden : "$hidden, "batch : "$BATCH_SIZE

            python scripts/train_fixed_encoder.py --psf $psf --down_orig $down_orig \
            --batch_size $BATCH_SIZE --n_epoch $N_EPOCH --opti adam --object_height $OBJECT_HEIGHT \
            --mask2sensor $mask2sensor --noise_type poisson --seed $SEED --hidden $hidden \
            --down_psf $down_psf --device cuda:0 --crop_psf $crop_psf --n_files $N_FILES --perspective 0.5


        done

        if [ "$TRAIN_HYBRID" = true ] ; then
            printf "\n------- SHIFT\n"
            down_psf=$DOWN_PSF_DEFAULT
            crop_psf=0
            mask2sensor=0.004
            echo "hybrid", "down_orig : "$down_orig, "mask2sensor : "$mask2sensor, "down_psf : "$down_psf, "crop_psf : "$crop_psf, "hidden : "$hidden, "batch : "$BATCH_SIZE

            python scripts/train_hybrid.py --down_orig $down_orig --sensor_act relu --crop_fact 0.8 \
            --batch_size $BATCH_SIZE --n_epoch $N_EPOCH --opti adam --object_height $OBJECT_HEIGHT \
            --mask2sensor $mask2sensor --noise_type poisson --seed $SEED \
            --down_psf $down_psf --device cuda:0 --n_files $N_FILES --shift

            printf "\n------- RESCALE\n"

            python scripts/train_hybrid.py --down_orig $down_orig --sensor_act relu --crop_fact 0.8 \
            --batch_size $BATCH_SIZE --n_epoch $N_EPOCH --opti adam \
            --mask2sensor $mask2sensor --noise_type poisson --seed $SEED \
            --down_psf $down_psf --device cuda:0 --n_files $N_FILES --random_height 2 20

            printf "\n------- ROTATE\n"

            python scripts/train_hybrid.py --down_orig $down_orig --sensor_act relu --crop_fact 0.8 \
            --batch_size $BATCH_SIZE --n_epoch $N_EPOCH --opti adam --object_height $OBJECT_HEIGHT \
            --mask2sensor $mask2sensor --noise_type poisson --seed $SEED \
            --down_psf $down_psf --device cuda:0 --n_files $N_FILES --rotate 90

            printf "\n------- PERSPECTIVE\n"

            python scripts/train_hybrid.py --down_orig $down_orig --sensor_act relu --crop_fact 0.8 \
            --batch_size $BATCH_SIZE --n_epoch $N_EPOCH --opti adam --object_height $OBJECT_HEIGHT \
            --mask2sensor $mask2sensor --noise_type poisson --seed $SEED \
            --down_psf $down_psf --device cuda:0 --n_files $N_FILES --perspective 0.5
        fi

    done

done
