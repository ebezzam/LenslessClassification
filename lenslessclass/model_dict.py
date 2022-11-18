import pathlib as plib


model_dict = {
    "MNIST": {
        "Lens": {
            "LR": {
                "Original": {
                    "24x32": plib.Path(
                        "MNIST_lens_outdim768_height0.12_scene2mask0.4_poisson40.0_croppsf100_50epoch_seed0_MultiClassLogistic_18102022_00h47"
                    ),
                    "12x16": plib.Path(
                        "MNIST_lens_outdim192_height0.12_scene2mask0.4_poisson40.0_croppsf100_50epoch_seed0_MultiClassLogistic_18102022_02h07"
                    ),
                    "6x8": plib.Path(
                        "MNIST_lens_outdim48_height0.12_scene2mask0.4_poisson40.0_croppsf100_50epoch_seed0_MultiClassLogistic_18102022_03h21"
                    ),
                    "3x4": plib.Path(
                        "MNIST_lens_outdim12_height0.12_scene2mask0.4_poisson40.0_croppsf100_50epoch_seed0_MultiClassLogistic_18102022_04h33"
                    ),
                },
            },
            "FCNN": {
                "Original": {
                    "24x32": plib.Path(
                        "MNIST_lens_outdim768_height0.12_scene2mask0.4_poisson40.0_croppsf100_50epoch_batch32_seed0_SingleHidden800_03112022_19h35"
                    ),
                    "12x16": plib.Path(
                        "MNIST_lens_outdim192_height0.12_scene2mask0.4_poisson40.0_croppsf100_50epoch_batch32_seed0_SingleHidden800_03112022_21h26"
                    ),
                    "6x8": plib.Path(
                        "MNIST_lens_outdim48_height0.12_scene2mask0.4_poisson40.0_croppsf100_50epoch_batch32_seed0_SingleHidden800_03112022_23h13"
                    ),
                    "3x4": plib.Path(
                        "MNIST_lens_outdim12_height0.12_scene2mask0.4_poisson40.0_croppsf100_50epoch_sched20_batch32_seed0_SingleHidden800_04112022_12h01"
                    ),
                },
                "Shift": {
                    "24x32": plib.Path(
                        "MNIST_lens_outdim768_height0.12_scene2mask0.4_poisson40.0_croppsf100_RandomShift_50epoch_sched20_batch32_seed0_SingleHidden800_04112022_17h28"
                    ),
                },
                "Rescale": {
                    "24x32": plib.Path(
                        "MNIST_lens_outdim768_height0.02-0.2_scene2mask0.4_poisson40.0_croppsf100_50epoch_sched20_batch32_seed0_SingleHidden800_04112022_17h50"
                    ),
                },
                "Rotate": {
                    "24x32": plib.Path(
                        "MNIST_lens_outdim768_height0.12_scene2mask0.4_poisson40.0_croppsf100_RandomRotate90.0_50epoch_sched20_batch32_seed0_SingleHidden800_04112022_18h13"
                    ),
                },
                "Perspective": {
                    "24x32": plib.Path(
                        "MNIST_lens_outdim768_height0.12_scene2mask0.4_poisson40.0_croppsf100_RandomPerspective0.5_50epoch_sched20_batch32_seed0_SingleHidden800_04112022_18h35"
                    ),
                },
            },
        },
        "Coded aperture": {
            "LR": {
                "Original": {
                    "24x32": plib.Path(
                        "MNIST_simulated_mls63_mask2sensor0p0005_17052022_18h00_12bit_outdim768_height0.12_scene2mask0.4_poisson40.0_downpsf8.0_50epoch_seed0_MultiClassLogistic_18102022_00h30"
                    ),
                    "12x16": plib.Path(
                        "MNIST_simulated_mls63_mask2sensor0p0005_17052022_18h00_12bit_outdim192_height0.12_scene2mask0.4_poisson40.0_downpsf8.0_50epoch_seed0_MultiClassLogistic_18102022_01h51"
                    ),
                    "6x8": plib.Path(
                        "MNIST_simulated_mls63_mask2sensor0p0005_17052022_18h00_12bit_outdim48_height0.12_scene2mask0.4_poisson40.0_downpsf8.0_50epoch_seed0_MultiClassLogistic_18102022_03h07"
                    ),
                    "3x4": plib.Path(
                        "MNIST_simulated_mls63_mask2sensor0p0005_17052022_18h00_12bit_outdim12_height0.12_scene2mask0.4_poisson40.0_downpsf8.0_50epoch_seed0_MultiClassLogistic_18102022_04h19"
                    ),
                },
            },
            "FCNN": {
                "Original": {
                    "24x32": plib.Path(
                        "MNIST_simulated_mls63_mask2sensor0p0005_17052022_18h00_12bit_outdim768_height0.12_scene2mask0.4_poisson40.0_downpsf8.0_50epoch_batch32_seed0_SingleHidden800_03112022_19h12"
                    ),
                    "12x16": plib.Path(
                        "MNIST_simulated_mls63_mask2sensor0p0005_17052022_18h00_12bit_outdim192_height0.12_scene2mask0.4_poisson40.0_downpsf8.0_50epoch_batch32_seed0_SingleHidden800_03112022_21h04"
                    ),
                    "6x8": plib.Path(
                        "MNIST_simulated_mls63_mask2sensor0p0005_17052022_18h00_12bit_outdim48_height0.12_scene2mask0.4_poisson40.0_downpsf8.0_50epoch_batch32_seed0_SingleHidden800_03112022_22h52"
                    ),
                    "3x4": plib.Path(
                        "MNIST_simulated_mls63_mask2sensor0p0005_17052022_18h00_12bit_outdim12_height0.12_scene2mask0.4_poisson40.0_downpsf8.0_50epoch_sched20_batch32_seed0_SingleHidden800_04112022_11h40"
                    ),
                },
                "Shift": {
                    "24x32": plib.Path(
                        "MNIST_simulated_mls63_mask2sensor0p0005_17052022_18h00_12bit_outdim768_height0.12_scene2mask0.4_poisson40.0_downpsf8.0_RandomShift_50epoch_sched20_batch32_seed0_SingleHidden800_04112022_15h56"
                    ),
                },
                "Rescale": {
                    "24x32": plib.Path(
                        "MNIST_simulated_mls63_mask2sensor0p0005_17052022_18h00_12bit_outdim768_height0.02-0.2_scene2mask0.4_poisson40.0_downpsf8.0_50epoch_sched20_batch32_seed0_SingleHidden800_04112022_16h19"
                    ),
                },
                "Rotate": {
                    "24x32": plib.Path(
                        "MNIST_simulated_mls63_mask2sensor0p0005_17052022_18h00_12bit_outdim768_height0.12_scene2mask0.4_poisson40.0_downpsf8.0_RandomRotate90.0_50epoch_sched20_batch32_seed0_SingleHidden800_04112022_16h42"
                    ),
                },
                "Perspective": {
                    "24x32": plib.Path(
                        "MNIST_simulated_mls63_mask2sensor0p0005_17052022_18h00_12bit_outdim768_height0.12_scene2mask0.4_poisson40.0_downpsf8.0_RandomPerspective0.5_50epoch_sched20_batch32_seed0_SingleHidden800_04112022_17h05"
                    ),
                },
            },
        },
        "Diffuser": {
            "LR": {
                "Original": {
                    "24x32": plib.Path(
                        "MNIST_tape_outdim768_height0.12_scene2mask0.4_poisson40.0_downpsf8.0_50epoch_seed0_MultiClassLogistic_17102022_23h55"
                    ),
                    "12x16": plib.Path(
                        "MNIST_tape_outdim192_height0.12_scene2mask0.4_poisson40.0_downpsf8.0_50epoch_seed0_MultiClassLogistic_18102022_01h19"
                    ),
                    "6x8": plib.Path(
                        "MNIST_tape_outdim48_height0.12_scene2mask0.4_poisson40.0_downpsf8.0_50epoch_seed0_MultiClassLogistic_18102022_02h37"
                    ),
                    "3x4": plib.Path(
                        "MNIST_tape_outdim12_height0.12_scene2mask0.4_poisson40.0_downpsf8.0_50epoch_seed0_MultiClassLogistic_18102022_03h51"
                    ),
                },
            },
            "FCNN": {
                "Original": {
                    "24x32": plib.Path(
                        "MNIST_tape_outdim768_height0.12_scene2mask0.4_poisson40.0_downpsf8.0_50epoch_batch32_seed0_SingleHidden800_03112022_18h23"
                    ),
                    "12x16": plib.Path(
                        "MNIST_tape_outdim192_height0.12_scene2mask0.4_poisson40.0_downpsf8.0_50epoch_batch32_seed0_SingleHidden800_03112022_20h20"
                    ),
                    "6x8": plib.Path(
                        "MNIST_tape_outdim48_height0.12_scene2mask0.4_poisson40.0_downpsf8.0_50epoch_batch32_seed0_SingleHidden800_03112022_22h09"
                    ),
                    "3x4": plib.Path(
                        "MNIST_tape_outdim12_height0.12_scene2mask0.4_poisson40.0_downpsf8.0_50epoch_sched20_batch32_seed0_SingleHidden800_04112022_10h58"
                    ),
                },
                "Shift": {
                    "24x32": plib.Path(
                        "MNIST_tape_outdim768_height0.12_scene2mask0.4_poisson40.0_downpsf8.0_RandomShift_50epoch_sched20_batch32_seed0_SingleHidden800_04112022_11h12"
                    ),
                },
                "Rescale": {
                    "24x32": plib.Path(
                        "MNIST_tape_outdim768_height0.02-0.2_scene2mask0.4_poisson40.0_downpsf8.0_50epoch_sched20_batch32_seed0_SingleHidden800_04112022_11h35"
                    ),
                },
                "Rotate": {
                    "24x32": plib.Path(
                        "MNIST_tape_outdim768_height0.12_scene2mask0.4_poisson40.0_downpsf8.0_RandomRotate90.0_50epoch_sched20_batch32_seed0_SingleHidden800_04112022_11h59"
                    ),
                },
                "Perspective": {
                    "24x32": plib.Path(
                        "MNIST_tape_outdim768_height0.12_scene2mask0.4_poisson40.0_downpsf8.0_RandomPerspective0.5_50epoch_sched20_batch32_seed0_SingleHidden800_04112022_12h23"
                    ),
                },
            },
        },
        "Fixed mask (m)": {
            "LR": {
                "Original": {
                    "24x32": plib.Path(
                        "MNIST_adafruit_outdim768_height0.12_scene2mask0.4_poisson40.0_downpsf8.0_50epoch_seed0_MultiClassLogistic_18102022_00h13"
                    ),
                    "12x16": plib.Path(
                        "MNIST_adafruit_outdim192_height0.12_scene2mask0.4_poisson40.0_downpsf8.0_50epoch_seed0_MultiClassLogistic_18102022_01h35"
                    ),
                    "6x8": plib.Path(
                        "MNIST_adafruit_outdim48_height0.12_scene2mask0.4_poisson40.0_downpsf8.0_50epoch_seed0_MultiClassLogistic_18102022_02h52"
                    ),
                    "3x4": plib.Path(
                        "MNIST_adafruit_outdim12_height0.12_scene2mask0.4_poisson40.0_downpsf8.0_50epoch_seed0_MultiClassLogistic_18102022_04h05"
                    ),
                },
            },
            "FCNN": {
                "Original": {
                    "24x32": plib.Path(
                        "MNIST_adafruit_outdim768_height0.12_scene2mask0.4_poisson40.0_downpsf8.0_50epoch_batch32_seed0_SingleHidden800_03112022_18h48"
                    ),
                    "12x16": plib.Path(
                        "MNIST_adafruit_outdim192_height0.12_scene2mask0.4_poisson40.0_downpsf8.0_50epoch_batch32_seed0_SingleHidden800_03112022_20h42"
                    ),
                    "6x8": plib.Path(
                        "MNIST_adafruit_outdim48_height0.12_scene2mask0.4_poisson40.0_downpsf8.0_50epoch_batch32_seed0_SingleHidden800_03112022_22h31"
                    ),
                    "3x4": plib.Path(
                        "MNIST_adafruit_outdim12_height0.12_scene2mask0.4_poisson40.0_downpsf8.0_50epoch_sched20_batch32_seed0_SingleHidden800_04112022_11h19"
                    ),
                },
                "Shift": {
                    "24x32": plib.Path(
                        "MNIST_adafruit_outdim768_height0.12_scene2mask0.4_poisson40.0_downpsf8.0_RandomShift_50epoch_sched20_batch32_seed0_SingleHidden800_04112022_12h47"
                    ),
                },
                "Rescale": {
                    "24x32": plib.Path(
                        "MNIST_adafruit_outdim768_height0.02-0.2_scene2mask0.4_poisson40.0_downpsf8.0_50epoch_sched20_batch32_seed0_SingleHidden800_04112022_13h11"
                    ),
                },
                "Rotate": {
                    "24x32": plib.Path(
                        "MNIST_adafruit_outdim768_height0.12_scene2mask0.4_poisson40.0_downpsf8.0_RandomRotate90.0_50epoch_sched20_batch32_seed0_SingleHidden800_04112022_13h34"
                    ),
                },
                "Perspective": {
                    "24x32": plib.Path(
                        "MNIST_adafruit_outdim768_height0.12_scene2mask0.4_poisson40.0_downpsf8.0_RandomPerspective0.5_50epoch_sched20_batch32_seed0_SingleHidden800_04112022_13h58"
                    ),
                },
            },
        },
        "Fixed mask (s)": {
            "LR": {
                "Original": {
                    "24x32": plib.Path(
                        "MNIST_simulated_adafruit_deadspaceTrue_15052022_21h04_outdim768_height0.12_scene2mask0.4_poisson40.0_downpsf8.0_50epoch_seed0_MultiClassLogistic_17102022_23h38"
                    ),
                    "12x16": plib.Path(
                        "MNIST_simulated_adafruit_deadspaceTrue_15052022_21h04_outdim192_height0.12_scene2mask0.4_poisson40.0_downpsf8.0_50epoch_seed0_MultiClassLogistic_18102022_01h03"
                    ),
                    "6x8": plib.Path(
                        "MNIST_simulated_adafruit_deadspaceTrue_15052022_21h04_outdim48_height0.12_scene2mask0.4_poisson40.0_downpsf8.0_50epoch_seed0_MultiClassLogistic_18102022_02h22"
                    ),
                    "3x4": plib.Path(
                        "MNIST_simulated_adafruit_deadspaceTrue_15052022_21h04_outdim12_height0.12_scene2mask0.4_poisson40.0_downpsf8.0_50epoch_seed0_MultiClassLogistic_18102022_03h36"
                    ),
                },
            },
            "FCNN": {
                "Original": {
                    "24x32": plib.Path(
                        "MNIST_simulated_adafruit_deadspaceTrue_15052022_21h04_outdim768_height0.12_scene2mask0.4_poisson40.0_downpsf8.0_50epoch_batch32_seed0_SingleHidden800_03112022_17h58"
                    ),
                    "12x16": plib.Path(
                        "MNIST_simulated_adafruit_deadspaceTrue_15052022_21h04_outdim192_height0.12_scene2mask0.4_poisson40.0_downpsf8.0_50epoch_batch32_seed0_SingleHidden800_03112022_19h58"
                    ),
                    "6x8": plib.Path(
                        "MNIST_simulated_adafruit_deadspaceTrue_15052022_21h04_outdim48_height0.12_scene2mask0.4_poisson40.0_downpsf8.0_50epoch_batch32_seed0_SingleHidden800_03112022_21h48"
                    ),
                    "3x4": plib.Path(
                        "MNIST_simulated_adafruit_deadspaceTrue_15052022_21h04_outdim12_height0.12_scene2mask0.4_poisson40.0_downpsf8.0_50epoch_sched20_batch32_seed0_SingleHidden800_04112022_10h36"
                    ),
                },
                "Shift": {
                    "24x32": plib.Path(
                        "MNIST_simulated_adafruit_deadspaceTrue_15052022_21h04_outdim768_height0.12_scene2mask0.4_poisson40.0_downpsf8.0_RandomShift_50epoch_sched20_batch32_seed0_SingleHidden800_04112022_14h22"
                    ),
                },
                "Rescale": {
                    "24x32": plib.Path(
                        "MNIST_simulated_adafruit_deadspaceTrue_15052022_21h04_outdim768_height0.02-0.2_scene2mask0.4_poisson40.0_downpsf8.0_50epoch_sched20_batch32_seed0_SingleHidden800_04112022_14h45"
                    ),
                },
                "Rotate": {
                    "24x32": plib.Path(
                        "MNIST_simulated_adafruit_deadspaceTrue_15052022_21h04_outdim768_height0.12_scene2mask0.4_poisson40.0_downpsf8.0_RandomRotate90.0_50epoch_sched20_batch32_seed0_SingleHidden800_04112022_15h09"
                    ),
                },
                "Perspective": {
                    "24x32": plib.Path(
                        "MNIST_simulated_adafruit_deadspaceTrue_15052022_21h04_outdim768_height0.12_scene2mask0.4_poisson40.0_downpsf8.0_RandomPerspective0.5_50epoch_sched20_batch32_seed0_SingleHidden800_04112022_15h32"
                    ),
                },
            },
        },
        "Learned mask": {
            "LR": {
                "Original": {
                    "24x32": plib.Path(
                        "MNIST_no_psf_down8_height0.12_NORM_outdim768_50epoch_schedNone_batch100_seed0_SLM_MultiClassLogistic_poisson40.0_DSresize_08112022_12h21"
                    ),
                    "12x16": plib.Path(
                        "MNIST_no_psf_down8_height0.12_outdim192_50epoch_batch100_seed0_SLM_MultiClassLogistic_poisson40.0_DSresize_19102022_06h23"
                    ),
                    "6x8": plib.Path(
                        "MNIST_no_psf_down8_height0.12_outdim48_50epoch_batch100_seed0_SLM_MultiClassLogistic_poisson40.0_DSresize_19102022_06h24"
                    ),
                    "3x4": plib.Path(
                        "MNIST_no_psf_down8_height0.12_outdim12_50epoch_batch100_seed0_SLM_MultiClassLogistic_poisson40.0_DSresize_18102022_20h40"
                    ),
                },
            },
            "FCNN": {
                "Original": {
                    "24x32": plib.Path(
                        "MNIST_no_psf_down8_height0.12_NORM_outdim768_50epoch_sched20_batch64_seed0_SLM_SingleHidden800_poisson40.0_DSresize_03112022_14h01"
                    ),
                    "12x16": plib.Path(
                        "MNIST_no_psf_down8_height0.12_NORM_outdim192_50epoch_sched20_batch64_seed0_SLM_SingleHidden800_poisson40.0_DSresize_03112022_18h09"
                    ),
                    "6x8": plib.Path(
                        "MNIST_no_psf_down8_height0.12_NORM_outdim48_50epoch_sched20_batch64_seed0_SLM_SingleHidden800_poisson40.0_DSresize_03112022_23h21"
                    ),
                    "3x4": plib.Path(
                        "MNIST_no_psf_down8_height0.12_NORM_outdim12_50epoch_sched20_batch64_seed0_SLM_SingleHidden800_poisson40.0_DSresize_04112022_04h12"
                    ),
                },
                "Shift": {
                    "24x32": plib.Path(
                        "MNIST_no_psf_down8_height0.12_RandomShift_outdim768_50epoch_sched20_batch64_seed0_SLM_SingleHidden800_poisson40.0_DSresize_04112022_11h18"
                    ),
                },
                "Rescale": {
                    "24x32": plib.Path(
                        "MNIST_no_psf_down8_height0.02-0.2_outdim768_50epoch_sched20_batch64_seed0_SLM_SingleHidden800_poisson40.0_DSresize_04112022_16h29"
                    ),
                },
                "Rotate": {
                    "24x32": plib.Path(
                        "MNIST_no_psf_down8_height0.12_RandomRotate90.0_outdim768_50epoch_sched20_batch64_seed0_SLM_SingleHidden800_poisson40.0_DSresize_04112022_21h25"
                    ),
                },
                "Perspective": {
                    "24x32": plib.Path(
                        "MNIST_no_psf_down8_height0.12_RandomPerspective0.5_outdim768_50epoch_sched20_batch64_seed0_SLM_SingleHidden800_poisson40.0_DSresize_05112022_01h58"
                    ),
                },
            },
        },
    },
    "Smiling": {
        "Lens": {
            "FCNN": {
                "24x32": "Smiling_celeba_lens_outdim768_height0.27_scene2mask0.55_poisson40.0_croppsf100_100000files_50epoch_batch64_seed0_SingleHidden800_22102022_18h21",
                "3x4": "Smiling_celeba_lens_outdim12_height0.27_scene2mask0.55_poisson40.0_croppsf100_100000files_50epoch_batch64_seed0_SingleHidden800_22102022_18h56",
            },
        },
        "Coded aperture": {
            "FCNN": {
                "24x32": "Smiling_celeba_simulated_mls63_mask2sensor0p0005_17052022_18h00_12bit_outdim768_height0.27_scene2mask0.55_poisson40.0_downpsf8.0_NORM_100000files_50epoch_batch64_seed0_SingleHidden800_22102022_15h11",
                "3x4": "Smiling_celeba_simulated_mls63_mask2sensor0p0005_17052022_18h00_12bit_outdim12_height0.27_scene2mask0.55_poisson40.0_downpsf8.0_NORM_100000files_50epoch_batch64_seed0_SingleHidden800_22102022_15h36",
            },
        },
        "Diffuser": {
            "FCNN": {
                "24x32": "Smiling_celeba_tape_outdim768_height0.27_scene2mask0.55_poisson40.0_downpsf8.0_NORM_100000files_50epoch_batch64_seed0_SingleHidden800_22102022_16h08",
                "3x4": "Smiling_celeba_tape_outdim12_height0.27_scene2mask0.55_poisson40.0_downpsf8.0_NORM_100000files_50epoch_batch64_seed0_SingleHidden800_22102022_16h42",
            },
        },
        "Fixed mask (m)": {
            "FCNN": {
                "24x32": "Smiling_celeba_adafruit_outdim768_height0.27_scene2mask0.55_poisson40.0_downpsf8.0_NORM_100000files_50epoch_batch64_seed0_SingleHidden800_22102022_14h25",
                "3x4": "Smiling_celeba_adafruit_outdim12_height0.27_scene2mask0.55_poisson40.0_downpsf8.0_NORM_100000files_50epoch_batch64_seed0_SingleHidden800_22102022_14h50",
            },
        },
        "Fixed mask (s)": {
            "FCNN": {
                "24x32": "Smiling_celeba_simulated_adafruit_deadspaceTrue_15052022_21h04_outdim768_height0.27_scene2mask0.55_poisson40.0_downpsf8.0_NORM_100000files_50epoch_batch64_seed0_SingleHidden800_22102022_17h13",
                "3x4": "Smiling_celeba_simulated_adafruit_deadspaceTrue_15052022_21h04_outdim12_height0.27_scene2mask0.55_poisson40.0_downpsf8.0_NORM_100000files_50epoch_batch64_seed0_SingleHidden800_22102022_17h49",
            },
        },
        "Learned mask": {
            "FCNN": {
                "24x32": "celeba_no_psf_down8_height0.27_NORM_100000files_scene2mask0.55_outdim768_Smiling_50epoch_seed0_SLM_SingleHidden800_poisson40.0_22102022_07h16",
                "3x4": "celeba_no_psf_down8_height0.27_NORM_100000files_scene2mask0.55_outdim12_Smiling_50epoch_seed0_SLM_SingleHidden800_poisson40.0_22102022_13h55",
            },
        },
    },
    ### GENDER
    "Gender": {
        "Lens": {
            "FCNN": {
                # "24x32": "Male_celeba_lens_outdim768_height0.27_scene2mask0.55_poisson40.0_croppsf100_100000files_50epoch_batch64_seed0_SingleHidden800_22102022_17h02",
                # drop 0.5 to avoid overfitting
                "24x32": "Male_celeba_lens_outdim768_height0.27_scene2mask0.55_poisson40.0_croppsf100_100000files_50epoch_batch64_seed0_SingleHidden800drop0.5_22102022_17h18",
                "3x4": "Male_celeba_adafruit_outdim12_height0.27_scene2mask0.55_poisson40.0_downpsf8.0_NORM_100000files_50epoch_batch64_seed0_SingleHidden800_22102022_00h43",
            }
        },
        "Coded aperture": {
            "FCNN": {
                "24x32": "Male_celeba_simulated_mls63_mask2sensor0p0005_17052022_18h00_12bit_outdim768_height0.27_scene2mask0.55_poisson40.0_downpsf8.0_NORM_100000files_50epoch_batch64_seed0_SingleHidden800_22102022_14h18",
                "3x4": "Male_celeba_simulated_mls63_mask2sensor0p0005_17052022_18h00_12bit_outdim12_height0.27_scene2mask0.55_poisson40.0_downpsf8.0_NORM_100000files_50epoch_batch64_seed0_SingleHidden800_22102022_14h43",
            }
        },
        "Diffuser": {
            "FCNN": {
                "24x32": "Male_celeba_tape_outdim768_height0.27_scene2mask0.55_poisson40.0_downpsf8.0_NORM_100000files_50epoch_batch64_seed0_SingleHidden800_22102022_15h04",
                "3x4": "Male_celeba_tape_outdim12_height0.27_scene2mask0.55_poisson40.0_downpsf8.0_NORM_100000files_50epoch_batch64_seed0_SingleHidden800_22102022_15h28",
            }
        },
        "Fixed mask (m)": {
            "FCNN": {
                "24x32": plib.Path(
                    "Male_celeba_adafruit_outdim768_height0.27_scene2mask0.55_poisson40.0_downpsf8.0_NORM_100000files_50epoch_batch64_seed0_SingleHidden800_21102022_23h52"
                ),
                "3x4": plib.Path(
                    "Male_celeba_adafruit_outdim12_height0.27_scene2mask0.55_poisson40.0_downpsf8.0_NORM_100000files_50epoch_batch64_seed0_SingleHidden800_22102022_00h43"
                ),
            },
        },
        "Fixed mask (s)": {
            "FCNN": {
                "24x32": "Male_celeba_simulated_adafruit_deadspaceTrue_15052022_21h04_outdim768_height0.27_scene2mask0.55_poisson40.0_downpsf8.0_NORM_100000files_50epoch_batch64_seed0_SingleHidden800_22102022_15h57",
                "3x4": "Male_celeba_simulated_adafruit_deadspaceTrue_15052022_21h04_outdim12_height0.27_scene2mask0.55_poisson40.0_downpsf8.0_NORM_100000files_50epoch_batch64_seed0_SingleHidden800_22102022_16h32",
            }
        },
        "Learned mask": {
            "FCNN": {
                "24x32": plib.Path(
                    "celeba_no_psf_down8_height0.27_NORM_100000files_scene2mask0.55_outdim768_Male_50epoch_seed0_SLM_SingleHidden800_poisson40.0_21102022_17h11"
                ),
                "3x4": plib.Path(
                    "celeba_no_psf_down8_height0.27_NORM_100000files_scene2mask0.55_outdim12_Male_50epoch_seed0_SLM_SingleHidden800_poisson40.0_21102022_17h22"
                ),
                # multiple seeds
                "24x32_multi": {
                    0: "celeba_no_psf_down8_height0.27_NORM_100000files_scene2mask0.55_outdim768_Male_25epoch_seed0_SLM_SingleHidden800_poisson40.0_22102022_16h24",
                    1: "celeba_no_psf_down8_height0.27_NORM_100000files_scene2mask0.55_outdim768_Male_25epoch_seed1_SLM_SingleHidden800_poisson40.0_22102022_20h35",
                    2: "celeba_no_psf_down8_height0.27_NORM_100000files_scene2mask0.55_outdim768_Male_25epoch_seed2_SLM_SingleHidden800_poisson40.0_22102022_23h35",
                    3: "celeba_no_psf_down8_height0.27_NORM_100000files_scene2mask0.55_outdim768_Male_25epoch_seed3_SLM_SingleHidden800_poisson40.0_23102022_02h33",
                    4: "celeba_no_psf_down8_height0.27_NORM_100000files_scene2mask0.55_outdim768_Male_25epoch_seed4_SLM_SingleHidden800_poisson40.0_23102022_05h36",
                    5: "celeba_no_psf_down8_height0.27_NORM_100000files_scene2mask0.55_outdim768_Male_25epoch_seed5_SLM_SingleHidden800_poisson40.0_23102022_08h36",
                    6: "celeba_no_psf_down8_height0.27_NORM_100000files_scene2mask0.55_outdim768_Male_25epoch_seed6_SLM_SingleHidden800_poisson40.0_23102022_12h11",
                    7: "celeba_no_psf_down8_height0.27_NORM_100000files_scene2mask0.55_outdim768_Male_25epoch_seed7_SLM_SingleHidden800_poisson40.0_23102022_15h33",
                    8: "celeba_no_psf_down8_height0.27_NORM_100000files_scene2mask0.55_outdim768_Male_25epoch_seed8_SLM_SingleHidden800_poisson40.0_23102022_19h11",
                    9: "celeba_no_psf_down8_height0.27_NORM_100000files_scene2mask0.55_outdim768_Male_25epoch_seed9_SLM_SingleHidden800_poisson40.0_23102022_23h59",
                },
            }
        },
    },
    "CIFAR10": {
        "Lens": {
            "VGG11": {
                "27x36": "CIFAR10_lens_outdim972_height0.25_scene2mask0.4_poisson40.0_croppsf100_rgb_NORM_50epoch_batch32_seed0_VGG11_24102022_03h51",
                "13x17": "CIFAR10_lens_outdim221_height0.25_scene2mask0.4_poisson40.0_croppsf100_rgb_NORM_50epoch_batch32_seed0_VGG11_27102022_17h45",
                "6x8": "CIFAR10_lens_outdim48_height0.25_scene2mask0.4_poisson40.0_croppsf100_rgb_NORM_50epoch_sched10_batch32_seed0_VGG11_09112022_11h16",
                "3x4": "CIFAR10_lens_outdim12_height0.25_scene2mask0.4_poisson40.0_croppsf100_rgb_NORM_50epoch_batch32_seed0_VGG11_24102022_06h07",
            },
        },
        "Coded aperture": {
            "VGG11": {
                "27x36": "CIFAR10_simulated_mls63_mask2sensor0p0005_17052022_18h00_12bit_outdim972_height0.25_scene2mask0.4_poisson40.0_downpsf8.0_rgb_NORM_50epoch_batch32_seed0_VGG11_28102022_20h03",
                "13x17": "CIFAR10_simulated_mls63_mask2sensor0p0005_17052022_18h00_12bit_outdim221_height0.25_scene2mask0.4_poisson40.0_downpsf8.0_rgb_NORM_50epoch_batch32_seed0_VGG11_28102022_10h56",
                "6x8": "CIFAR10_simulated_mls63_mask2sensor0p0005_17052022_18h00_12bit_outdim48_height0.25_scene2mask0.4_poisson40.0_downpsf8.0_rgb_NORM_50epoch_batch32_seed0_VGG11_28102022_13h06",
                "3x4": "CIFAR10_simulated_mls63_mask2sensor0p0005_17052022_18h00_12bit_outdim12_height0.25_scene2mask0.4_poisson40.0_downpsf8.0_rgb_NORM_50epoch_batch32_seed0_VGG11_24102022_08h53",
            }
        },
        "Diffuser": {
            "VGG11": {
                "27x36": plib.Path(
                    "CIFAR10_tape_outdim972_height0.25_scene2mask0.4_poisson40.0_downpsf8.0_rgb_NORM_50epoch_batch32_seed0_VGG11_28102022_21h40"
                ),
                "13x17": plib.Path(
                    "CIFAR10_tape_outdim221_height0.25_scene2mask0.4_poisson40.0_downpsf8.0_rgb_NORM_50epoch_batch32_seed0_VGG11_28102022_15h15"
                ),
                "6x8": plib.Path(
                    "CIFAR10_tape_outdim48_height0.25_scene2mask0.4_poisson40.0_downpsf8.0_rgb_NORM_50epoch_batch32_seed0_VGG11_28102022_17h14"
                ),
                "3x4": plib.Path(
                    "CIFAR10_tape_outdim12_height0.25_scene2mask0.4_poisson40.0_downpsf8.0_rgb_NORM_50epoch_batch32_seed0_VGG11_24102022_11h57"
                ),
            }
        },
        "Fixed mask (m)": {
            "VGG11": {
                "27x36": "CIFAR10_adafruit_outdim972_height0.25_scene2mask0.4_poisson40.0_downpsf8.0_rgb_NORM_50epoch_batch32_seed0_VGG11_24102022_00h23",
                "13x17": "CIFAR10_adafruit_outdim221_height0.25_scene2mask0.4_poisson40.0_downpsf8.0_rgb_NORM_50epoch_batch32_seed0_VGG11_27102022_20h49",
                "6x8": "CIFAR10_adafruit_outdim48_height0.25_scene2mask0.4_poisson40.0_downpsf8.0_rgb_NORM_50epoch_batch32_seed0_VGG11_28102022_08h46",
                "3x4": "CIFAR10_adafruit_outdim12_height0.25_scene2mask0.4_poisson40.0_downpsf8.0_rgb_NORM_50epoch_batch32_seed0_VGG11_24102022_01h36",
            },
        },
        "Fixed mask (s)": {
            "VGG11": {
                "27x36": "CIFAR10_simulated_adafruit_deadspaceTrue_15052022_21h04_outdim972_height0.25_scene2mask0.4_poisson40.0_downpsf8.0_rgb_NORM_50epoch_batch32_seed0_VGG11_28102022_23h13",
                "13x17": "CIFAR10_simulated_adafruit_deadspaceTrue_15052022_21h04_outdim221_height0.25_scene2mask0.4_poisson40.0_downpsf8.0_rgb_NORM_50epoch_batch32_seed0_VGG11_28102022_18h52",
                "6x8": "CIFAR10_simulated_adafruit_deadspaceTrue_15052022_21h04_outdim48_height0.25_scene2mask0.4_poisson40.0_downpsf8.0_rgb_NORM_50epoch_batch32_seed0_VGG11_28102022_20h36",
                "3x4": "CIFAR10_simulated_adafruit_deadspaceTrue_15052022_21h04_outdim12_height0.25_scene2mask0.4_poisson40.0_downpsf8.0_rgb_NORM_50epoch_batch32_seed0_VGG11_24102022_15h08",
            }
        },
        "Learned mask": {
            "VGG11": {
                "27x36": "CIFAR10_no_psf_down8_height0.25_NORM_outdim2916_50epoch_batch32_seed0_SLM_VGG11_poisson40.0_DSresize_24102022_11h47",
                "13x17": "CIFAR10_no_psf_down8_height0.25_NORM_outdim663_50epoch_sched20_batch32_seed0_SLM_VGG11_poisson40.0_DSresize_31102022_09h13",
                "6x8": "CIFAR10_no_psf_down8_height0.25_NORM_outdim144_50epoch_sched20_batch32_seed0_SLM_VGG11_poisson40.0_DSresize_31102022_16h22",
                "3x4": "CIFAR10_no_psf_down8_height0.25_NORM_outdim36_50epoch_batch32_seed0_SLM_VGG11_poisson40.0_DSresize_23102022_17h56",
            }
        },
    },
    "celeba_decoder": {
        "100": {
            "1 mask": "celeba_1_learned_mixed_mask_out768_offset100000_nfiles100000_50epoch_batch4_schedNone_seed0_Conv3_10000_l1_100trainfiles_27102022_14h22",
            "10 masks": "celeba_10_learned_mixed_mask_out768_offset100000_nfiles100000_50epoch_batch4_schedNone_seed0_Conv3_10000_l1_100trainfiles_27102022_21h17",
            "10 random": "celeba_10_random_mixed_mask_nonlinTrue_out768_offset100000_nfiles100000_50epoch_batch4_schedNone_seed0_Conv3_10000_l1_100trainfiles_28102022_03h53",
            "100 masks": "celeba_100_random_mixed_mask_nonlinTrue_out768_offset100000_nfiles100000_50epoch_batch4_schedNone_seed0_Conv3_10000_l1_100trainfiles_28102022_01h54",
        },
        "1000": {
            "1 mask": "celeba_1_learned_mixed_mask_out768_offset100000_nfiles100000_50epoch_batch16_schedNone_seed0_Conv3_10000_l1_1000trainfiles_27102022_19h54",
            "10 masks": "celeba_10_learned_mixed_mask_out768_offset100000_nfiles100000_50epoch_batch16_schedNone_seed0_Conv3_10000_l1_1000trainfiles_27102022_21h37",
            "10 random": "celeba_10_random_mixed_mask_nonlinTrue_out768_offset100000_nfiles100000_50epoch_batch16_schedNone_seed0_Conv3_10000_l1_1000trainfiles_28102022_04h00",
            "100 masks": "celeba_100_random_mixed_mask_nonlinTrue_out768_offset100000_nfiles100000_50epoch_batch16_schedNone_seed0_Conv3_10000_l1_1000trainfiles_28102022_02h09",
        },
        "10000": {
            "1 mask": "celeba_1_learned_mixed_mask_out768_offset100000_nfiles100000_50epoch_batch32_schedNone_seed0_Conv3_10000_l1_10000trainfiles_27102022_13h46",
            "10 masks": "celeba_10_learned_mixed_mask_out768_offset100000_nfiles100000_50epoch_batch32_schedNone_seed0_Conv3_10000_l1_10000trainfiles_27102022_22h23",
            "10 random": "celeba_10_random_mixed_mask_nonlinTrue_out768_offset100000_nfiles100000_50epoch_batch32_schedNone_seed0_Conv3_10000_l1_10000trainfiles_28102022_04h15",
            "100 masks": "celeba_100_random_mixed_mask_nonlinTrue_out768_offset100000_nfiles100000_50epoch_batch32_schedNone_seed0_Conv3_10000_l1_10000trainfiles_28102022_02h33",
        },
        "100000": {
            "1 mask": "celeba_1_learned_mixed_mask_out768_offset100000_nfiles100000_50epoch_batch32_schedNone_seed0_Conv3_10000_l1_100000trainfiles_26102022_09h41",
            "10 masks": "celeba_10_learned_mixed_mask_out768_offset100000_nfiles100000_50epoch_batch32_schedNone_seed0_Conv3_10000_l1_100000trainfiles_26102022_17h16",
            "10 random": "celeba_10_random_mixed_mask_nonlinTrue_out768_offset100000_nfiles100000_50epoch_batch32_schedNone_seed0_Conv3_10000_l1_100000trainfiles_28102022_05h35",
            "100 masks": "celeba_100_random_mixed_mask_nonlinTrue_out768_offset100000_nfiles100000_50epoch_batch32_schedNone_seed0_Conv3_10000_l1_100000trainfiles_27102022_19h46",
        },
    },
}
