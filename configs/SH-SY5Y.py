from pacmap.utils import crop_stack

dataset_name = "SH-SY5Y"

# Raw data info
config = {
    "raw": {
        "input_path": f"{dataset_name}/raw",
        "voxelsize": [1.9999, 0.3594, 0.3594]
    }
}

# Preprocessing
config.update({
    "preprocess": {
        "input_path": f"{dataset_name}/raw",
        "output_path": f"{dataset_name}/preprocessed",
        "output_path_binary": f"{dataset_name}/binary",
        "channels2store": [0],
        "current_voxelsize": config['raw']['voxelsize'], 
        "target_voxelsize": config['raw']['voxelsize'],
        'crop_func': crop_stack,
        "crop_channel": 0,
        "min_size": 0.03
    }
})

# Binarization
config.update({
    "binarize": {
        "input_path": f"{dataset_name}/preprocessed",
        "output_path": f"{dataset_name}/binary",
        "channel2use": 0
    },
})

# Patchify preprocessed images
config.update({
    "patch_creation": {
        "preprocessed": {
            "input_path": f"{dataset_name}/preprocessed",
            "output_path": f"{dataset_name}/patches",
            "patch_size": [46, 256, 256],
            "patch_stride": [40, 224, 224],
            "channels2store": [0],
        }
    }
})

# Patchify binary images
config['patch_creation'].update({
    "binarized": {
        "input_path": f"{dataset_name}/binary",
        "output_path": f"{dataset_name}/patches_binary",
        "patch_size": config['patch_creation']['preprocessed']['patch_size'],
        "patch_stride": config['patch_creation']['preprocessed']['patch_stride'],
        "channels2store": [0]
    }
})

# Generate weak targets
RADI_UM = [10, 10, 10]
config.update({
    "weak_targets": {
        "input_path_grayscale": f"{dataset_name}/test/patches",
        "input_path_binary": f"{dataset_name}/test/patches_binary",
        "output_path": f"{dataset_name}/test/model_pred/weak_targets",
        "radi_um": RADI_UM,
        "voxelsize": config['raw']['voxelsize'],
        "min_distance": 7.5,
        "exclude_border": True,
        "save_points_csv": True, # Stored in output_path/points_csv
        "save_targets": True # Stored in output_path/targets
    }
})

# Generate ground truth training targets
config.update({
    "points2prob": {
        "targets_cmap": {
            "input_path": f"{dataset_name}/train/points_csv",
            "output_path": f"{dataset_name}/train/targets_cmap",
            "img_path": f"{dataset_name}/train/patches",
            "radi_um": RADI_UM,
            "intensity_as_spacing": False,
            "voxelsize": config['raw']['voxelsize'],
            "method": "csv",
            "check_batches": True,
            "save_targets": True # Stored in output_path/targets
        },
        "targets_pacmap": {
            "input_path": f"{dataset_name}/train/points_csv",
            "output_path": f"{dataset_name}/train/targets_pacmap",
            "img_path": f"{dataset_name}/train/patches",
            "radi_um": RADI_UM,
            "intensity_as_spacing": True,
            "voxelsize": config['raw']['voxelsize'],
            "method": "csv",
            "check_batches": True,
            "save_targets": True # Stored in output_path/targets
        },
    }
})

# Train nuclei centroid prediction model
INPUT_PATH = f"{dataset_name}/train/patches"
SPLIT = [0.8, 0.2, 0.0]
BATCH_SIZE = 4
NUM_EPOCHS = 150
LR = 1e-4
MODEL_ARCH = "UNet3D"
LOSS = "MSE"
PATIENCE = 10
F_MAPS = 32
DEPTH = 4
CHECK_BATCHES = True
AUGMENT_RESCALE_P = 0.0
GPU_ID = 0
PRETRAIN_DATASET = "LN18-RED"

seeds = [0, 1, 2]
model_types = ["scratch", "finetuned"]
for SEED in seeds:
    for MODEL_TYPE in model_types:
        cmap_pretrained = f"{PRETRAIN_DATASET}/models/pretrain-{SEED}" if MODEL_TYPE == "finetuned" else None
        pacmap_pretrained = f"{PRETRAIN_DATASET}/models/pretrain-{SEED}-d" if MODEL_TYPE == "finetuned" else None
        config.update({
            "train": {
                f"cmap-{MODEL_TYPE}-{SEED}": {
                    "input_path": INPUT_PATH,
                    "target_path": f"{dataset_name}/train/targets_cmap/targets",
                    "output_path": f"{dataset_name}/models/cmap-{MODEL_TYPE}-{SEED}",
                    "pretrained": cmap_pretrained,
                    "gpu_id": GPU_ID,
                    "random_seed": SEED,
                    "final_sigmoid": True,
                    "normalize_targets": True,
                    "split": SPLIT,
                    "batch_size": BATCH_SIZE,
                    "num_epochs": NUM_EPOCHS,
                    "lr": LR,
                    "model_type": MODEL_ARCH,
                    "loss": LOSS,
                    "patience": PATIENCE,
                    "f_maps": F_MAPS,
                    "check_batches": CHECK_BATCHES,
                    "augment_rescale_p": AUGMENT_RESCALE_P,
                },
                f"pacmap-{MODEL_TYPE}-{SEED}": {
                    "input_path": INPUT_PATH,
                    "target_path": f"manual_annotation/{dataset_name}/train/targets_pacmap/targets",
                    "output_path": f"{dataset_name}/models/pacmap-{MODEL_TYPE}-{SEED}",
                    "pretrained": pacmap_pretrained,
                    "gpu_id": GPU_ID,
                    "random_seed": SEED,
                    "final_sigmoid": False,
                    "normalize_targets": False,
                    "split": SPLIT,
                    "batch_size": BATCH_SIZE,
                    "num_epochs": NUM_EPOCHS,
                    "lr": LR,
                    "model_type": MODEL_ARCH,
                    "loss": LOSS,
                    "patience": PATIENCE,
                    "f_maps": F_MAPS,
                    "check_batches": CHECK_BATCHES,
                    "augment_rescale_p": AUGMENT_RESCALE_P,
                },
            }
        })

# Prediction configs
INPUT_PATH = f"{dataset_name}/test/patches"
OUTPUT_PATH_BASE = f"{dataset_name}/test/model_pred"
MODEL_ARCH = "UNet3D"
F_MAPS = 32
DEPTH = 4
THRESHOLD_CMAP = 0.2
THRESHOLD_PACMAP = 2.5
MIN_DISTANCE = 5
CHECK_BATCHES = True
DO_BORDER = False
VOXELSIZE = config['raw']['voxelsize']
PRETRAIN_DATASET = "LN18-RED"
GPU_ID = 0

seeds = [0, 1, 2]
model_types = ["pretrain", "scratch", "finetuned"]
config['pred_model'] = {}
for SEED in seeds:
    for MODEL_TYPE in model_types:
        MODEL_PATH_BASE = f"{PRETRAIN_DATASET}/models" if MODEL_TYPE == "pretrained" else f"{dataset_name}/models"
        config['pred_model'].update({
            f"cmap-{MODEL_TYPE}-{SEED}":{
                "input_path":INPUT_PATH,
                "output_path": f"{OUTPUT_PATH_BASE}/cmap-{MODEL_TYPE}-{SEED}",
                "model_path": f"{MODEL_PATH_BASE}/cmap-{MODEL_TYPE}-{SEED}",
                "model_type": MODEL_ARCH,
                "final_sigmoid": True,
                "gpu_id": GPU_ID,
                "merge_close_points": False,
                "do_border":True,
                "threshold_abs": THRESHOLD_CMAP,
                "intensity_as_spacing": False,
                "top_down": True,
                "voxelsize": VOXELSIZE,
                "save_csv": True,
                "save_preds": True,
                "model_type": MODEL_TYPE,
                "f_maps": F_MAPS,
                "depth": DEPTH,
                "min_distance": MIN_DISTANCE,
                "do_border": DO_BORDER,
                "check_batches": CHECK_BATCHES
            },
            f"pacmap-{MODEL_TYPE}-{SEED}": {
                "input_path": INPUT_PATH,
                "output_path": f"{OUTPUT_PATH_BASE}/pacmap-{MODEL_TYPE}-{SEED}",
                "model_path": f"{MODEL_PATH_BASE}/pacmap-{MODEL_TYPE}-{SEED}",
                "final_sigmoid": False,
                "gpu_id": GPU_ID,
                "merge_close_points": False,
                "threshold_abs": THRESHOLD_PACMAP,
                "intensity_as_spacing": True,
                "top_down": False,
                "voxelsize": VOXELSIZE,
                "save_csv": True,
                "save_preds": True,
                "model_type": MODEL_ARCH,
                "f_maps": F_MAPS,
                "depth": DEPTH,
                "min_distance": MIN_DISTANCE,
                "do_border": DO_BORDER,
                "check_batches": CHECK_BATCHES
            },
        })


# Performance evaluation
TRUE_PATH = f"{dataset_name}/test/points_csv"
TRUE2POINTS_METHOD = "csv"
OUTPUT_PATH = f"{dataset_name}/test/model_pred"
VOXELSIZE = config['raw']['voxelsize']
VOLUMESIZE_VOX = [46, 256, 256]
BORDERSIZE_UM = 5
THRESHOLD_SAU = 0.4

config.update({
    "performance": {
        "seeded_watershed": {
            "true_path": TRUE_PATH,
            "true2points_method": TRUE2POINTS_METHOD,
            "pred_path": f"{dataset_name}/test/model_pred/weak_targets/points_csv",
            "pred2points_method": "csv",
            "output_path": OUTPUT_PATH,
            "filename": "seeded_watershed",
            "voxelsize": VOXELSIZE,
            "volumesize_vox": VOLUMESIZE_VOX,
            "bordersize_um": BORDERSIZE_UM
        },
        "stardist": {
            "true_path": TRUE_PATH,
            "true2points_method": TRUE2POINTS_METHOD,
            "pred_path": f"{dataset_name}/test/model_pred/stardist",
            "pred2points_method": "masks",
            "output_path": OUTPUT_PATH,
            "filename": "stardist",
            "voxelsize": VOXELSIZE,
            "volumesize_vox": VOLUMESIZE_VOX,
            "bordersize_um": BORDERSIZE_UM
        }
    }
})

model_types = ["pretrained", "scratch" "finetuned"]
seeds = [0, 1, 2]
for MODEL_TYPE in model_types:
    for SEED in seeds:
        config["performance"].update({
            f"sau-net-{MODEL_TYPE}-{SEED}": {
                "true_path": TRUE_PATH,
                "true2points_method": TRUE2POINTS_METHOD,
                "pred_path": f"{dataset_name}/test/model_pred/cmap-{MODEL_TYPE}-{SEED}/preds",
                "pred2points_method": "cc",
                "output_path": OUTPUT_PATH,
                "filename": f"sau-net-{MODEL_TYPE}-{SEED}",
                "voxelsize": VOXELSIZE,
                "volumesize_vox": VOLUMESIZE_VOX,
                "bordersize_um": BORDERSIZE_UM,
                "threshold": THRESHOLD_SAU
            },
            f"cmap-{MODEL_TYPE}-{SEED}": {
                "true_path": TRUE_PATH,
                "true2points_method": TRUE2POINTS_METHOD,
                "pred_path": f"{dataset_name}/test/model_pred/cmap-{MODEL_TYPE}-{SEED}/preds",
                "pred2points_method": "peaks",
                "output_path": OUTPUT_PATH,
                "filename": f"cmap-{MODEL_TYPE}-{SEED}",
                "voxelsize": VOXELSIZE,
                "volumesize_vox": VOLUMESIZE_VOX,
                "bordersize_um": BORDERSIZE_UM,
                "threshold": THRESHOLD_CMAP
            },
            f"pacmap-{MODEL_TYPE}-{SEED}": {
                "true_path": TRUE_PATH,
                "true2points_method": TRUE2POINTS_METHOD,
                "pred_path": f"{dataset_name}/test/model_pred/pacmap-{MODEL_TYPE}-{SEED}/preds",
                "pred2points_method": "peaks",
                "output_path": OUTPUT_PATH,
                "filename": f"pacmap-{MODEL_TYPE}-{SEED}",
                "voxelsize": VOXELSIZE,
                "volumesize_vox": VOLUMESIZE_VOX,
                "bordersize_um": BORDERSIZE_UM,
                "threshold": THRESHOLD_PACMAP,
                "intensity_as_spacing": True,
                "top_down": False
            }
        })