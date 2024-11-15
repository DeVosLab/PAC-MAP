import numpy as np

from pacmap.utils import crop_stack

dataset_name = "SH-SY5Y"

# RAW DATA INFO
config = {
    "raw": {
        "input_path": f"{dataset_name}/raw",
        "voxelsize": [1.9999, 0.3594, 0.3594]
    }
}

# PREPROCESSING
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

# BINARIZATION
config.update({
    "binarize": {
        "input_path": f"{dataset_name}/preprocessed",
        "output_path": f"{dataset_name}/binary",
        "channel2use": 0
    },
})

# PATCHIFY PREPROCESSED IMAGES
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

# PATCHIFY BINARY IMAGES
config['patch_creation'].update({
    "binarized": {
        "input_path": f"{dataset_name}/binary",
        "output_path": f"{dataset_name}/patches_binary",
        "patch_size": config['patch_creation']['preprocessed']['patch_size'],
        "patch_stride": config['patch_creation']['preprocessed']['patch_stride'],
        "channels2store": [0]
    }
})

# GENERATE WEAK TARGETS
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

# GENERATE GROUND TRUTH TRAINING TARGETS
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

# TRAIN STANDARD 3D U-NET BASED MODELS
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
config["train"] = {}
for SEED in seeds:
    for MODEL_TYPE in model_types:
        cmap_pretrained = f"{PRETRAIN_DATASET}/models/pretrain-{SEED}" if MODEL_TYPE == "finetuned" else None
        pacmap_pretrained = f"{PRETRAIN_DATASET}/models/pretrain-{SEED}-d" if MODEL_TYPE == "finetuned" else None

        config["train"].update({
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
        })

# TRAIN SAU-NET BASED MODELS
INPUT_PATH = f"{dataset_name}/train/patches"
SPLIT = [0.8, 0.2, 0.0]
BATCH_SIZE = 4
NUM_EPOCHS = 150
LR = 1e-4
ETA_MIN = LR * 1e-3
OPTIMIZER = "AdamW"
WEIGHT_DECAY = 1e-3
SCHEDULER = "CosineAnnealingWarmRestarts"
T_0 = 15
T_MULT = 2
MODEL_ARCH = "SAUNet3D"
LOSS = "MSE"
F_MAPS = 32
DEPTH = 4
CHECK_BATCHES = True
AUGMENT_RESCALE_P = 0.0
GPU_ID = 0
MODEL_TYPE = "scratch"

seeds = [0, 1, 2]
for SEED in seeds:
    config["train"].update({
        f"sau-cmap-{MODEL_TYPE}-{SEED}": {
            "input_path": INPUT_PATH,
            "target_path": f"{dataset_name}/train/targets_cmap/targets",
            "output_path": f"{dataset_name}/models/sau-cmap-{MODEL_TYPE}-{SEED}",
            "pretrained": None,
            "gpu_id": GPU_ID,
            "random_seed": SEED,
            "final_sigmoid": True,
            "normalize_targets": True,
            "split": SPLIT,
            "batch_size": BATCH_SIZE,
            "num_epochs": NUM_EPOCHS,
            "lr": LR,
            "optimizer": OPTIMIZER,
            "weight_decay": WEIGHT_DECAY,
            "scheduler": SCHEDULER,
            "T_0": T_0,
            "T_mult": T_MULT,
            "eta_min": ETA_MIN,
            "model_type": MODEL_ARCH,
            "loss": LOSS,
            "f_maps": F_MAPS,
            "check_batches": CHECK_BATCHES,
            "augment_rescale_p": AUGMENT_RESCALE_P,
        },
        f"sau-pacmap-{MODEL_TYPE}-{SEED}": {
            "input_path": INPUT_PATH,
            "target_path": f"manual_annotation/{dataset_name}/train/targets_pacmap/targets",
            "output_path": f"{dataset_name}/models/sau-pacmap-{MODEL_TYPE}-{SEED}",
            "pretrained": None,
            "gpu_id": GPU_ID,
            "random_seed": SEED,
            "final_sigmoid": False,
            "normalize_targets": False,
            "split": SPLIT,
            "batch_size": BATCH_SIZE,
            "num_epochs": NUM_EPOCHS,
            "lr": LR,
            "optimizer": OPTIMIZER,
            "weight_decay": WEIGHT_DECAY,
            "scheduler": SCHEDULER,
            "T_0": T_0,
            "T_mult": T_MULT,
            "eta_min": ETA_MIN,
            "model_type": MODEL_ARCH,
            "loss": LOSS,
            "f_maps": F_MAPS,
            "check_batches": CHECK_BATCHES,
            "augment_rescale_p": AUGMENT_RESCALE_P,
        },
    })

# PREDICT WITH STANDARD 3D U-NET BASED MODELS
INPUT_PATH = f"{dataset_name}/test/patches"
OUTPUT_PATH_BASE = f"{dataset_name}/test/model_pred"
MODEL_ARCH = "UNet3D"
F_MAPS = 32
DEPTH = 4
MIN_DISTANCE = 5
CHECK_BATCHES = True
VOXELSIZE = config['raw']['voxelsize']
PRETRAIN_DATASET = "LN18-RED"
GPU_ID = 0
SAVE_CSV = False
SAVE_PREDS = True

seeds = [0, 1, 2]
model_types = ["pretrain", "scratch", "finetuned"]
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
                "voxelsize": VOXELSIZE,
                "save_csv": SAVE_CSV,
                "save_preds": SAVE_PREDS,
                "model_type": MODEL_TYPE,
                "f_maps": F_MAPS,
                "depth": DEPTH,
                "check_batches": CHECK_BATCHES
            },
            f"pacmap-{MODEL_TYPE}-{SEED}": {
                "input_path": INPUT_PATH,
                "output_path": f"{OUTPUT_PATH_BASE}/pacmap-{MODEL_TYPE}-{SEED}",
                "model_path": f"{MODEL_PATH_BASE}/pacmap-{MODEL_TYPE}-{SEED}",
                "final_sigmoid": False,
                "gpu_id": GPU_ID,
                "voxelsize": VOXELSIZE,
                "save_csv": SAVE_CSV,
                "save_preds": SAVE_PREDS,
                "model_type": MODEL_ARCH,
                "f_maps": F_MAPS,
                "depth": DEPTH,
                "check_batches": CHECK_BATCHES
            },
        })


# PREDICT WITH SAU-NET BASED MODELS
INPUT_PATH = f"{dataset_name}/test/patches"
OUTPUT_PATH_BASE = f"{dataset_name}/test/model_pred"
MODEL_PATH_BASE = f"{dataset_name}/models"
MODEL_ARCH = "SAUNet3D"
F_MAPS = 32
DEPTH = 4
CHECK_BATCHES = True
VOXELSIZE = config['raw']['voxelsize']
GPU_ID = 0
MODEL_TYPE = "scratch"
SAVE_CSV = False
SAVE_PREDS = True

seeds = [0, 1, 2]
for SEED in seeds:
    config['pred_model'].update({
        f"sau-cmap-{MODEL_TYPE}-{SEED}":{
            "input_path":INPUT_PATH,
            "output_path": f"{OUTPUT_PATH_BASE}/sau-cmap-{MODEL_TYPE}-{SEED}",
            "model_path": f"{MODEL_PATH_BASE}/sau-cmap-{MODEL_TYPE}-{SEED}",
            "model_type": MODEL_ARCH,
            "final_sigmoid": True,
            "gpu_id": GPU_ID,
            "voxelsize": VOXELSIZE,
            "save_csv": SAVE_CSV,
            "save_preds": SAVE_PREDS,
            "model_type": MODEL_TYPE,
            "f_maps": F_MAPS,
            "depth": DEPTH,
            "check_batches": CHECK_BATCHES
        },
        f"sau-pacmap-{MODEL_TYPE}-{SEED}": {
            "input_path": INPUT_PATH,
            "output_path": f"{OUTPUT_PATH_BASE}/sau-pacmap-{MODEL_TYPE}-{SEED}",
            "model_path": f"{MODEL_PATH_BASE}/sau-pacmap-{MODEL_TYPE}-{SEED}",
            "final_sigmoid": False,
            "gpu_id": GPU_ID,
            "voxelsize": VOXELSIZE,
            "save_csv": SAVE_CSV,
            "save_preds": SAVE_PREDS,
            "model_type": MODEL_ARCH,
            "f_maps": F_MAPS,
            "depth": DEPTH,
            "check_batches": CHECK_BATCHES
        },
    })

# PERFORMANCE EVALUATION
TRUE_PATH = f"{dataset_name}/test/points_csv"
TRUE2POINTS_METHOD = "csv"
OUTPUT_PATH = f"{dataset_name}/test/model_pred"
VOXELSIZE = config['raw']['voxelsize']
VOLUMESIZE_VOX = [46, 256, 256]
BORDERSIZE_UM = 5
n_steps = 15
t_min = 0
t_max = 1
THRESHOLD_SEG = np.linspace(t_min, t_max, n_steps).tolist() # Segmentation threshold
THRESHOLD = np.linspace(t_min, t_max, n_steps).tolist() # Probability threshold
THRESHOLD_D = (11*np.linspace(t_min, t_max, n_steps)).tolist() # Distance threshold

# Seeded watershed and StarDist
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

# 3D U-Net based models
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
                "threshold": THRESHOLD
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
                "threshold": THRESHOLD
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
                "threshold": THRESHOLD_D,
                "intensity_as_spacing": True,
                "top_down": False
            }
        })

# SAU-Net based models
seeds = [0, 1, 2]
MODEL_TYPE = "scratch"
for SEED in seeds:
    config["performance"].update({
        f"sau-sau-net-{MODEL_TYPE}-{SEED}": {
            "true_path": TRUE_PATH,
            "true2points_method": TRUE2POINTS_METHOD,
            "pred_path": f"{dataset_name}/test/model_pred/sau-cmap-{MODEL_TYPE}-{SEED}/preds",
            "pred2points_method": "cc",
            "output_path": OUTPUT_PATH,
            "filename": f"sau-sau-net-{MODEL_TYPE}-{SEED}",
            "voxelsize": VOXELSIZE,
            "volumesize_vox": VOLUMESIZE_VOX,
            "bordersize_um": BORDERSIZE_UM,
            "threshold": THRESHOLD
        },
        f"sau-cmap-{MODEL_TYPE}-{SEED}": {
            "true_path": TRUE_PATH,
            "true2points_method": TRUE2POINTS_METHOD,
            "pred_path": f"{dataset_name}/test/model_pred/sau-cmap-{MODEL_TYPE}-{SEED}/preds",
            "pred2points_method": "peaks",
            "output_path": OUTPUT_PATH,
            "filename": f"sau-cmap-{MODEL_TYPE}-{SEED}",
            "voxelsize": VOXELSIZE,
            "volumesize_vox": VOLUMESIZE_VOX,
            "bordersize_um": BORDERSIZE_UM,
            "threshold": THRESHOLD
        },
        f"sau-pacmap-{MODEL_TYPE}-{SEED}": {
            "true_path": TRUE_PATH,
            "true2points_method": TRUE2POINTS_METHOD,
            "pred_path": f"{dataset_name}/test/model_pred/sau-pacmap-{MODEL_TYPE}-{SEED}/preds",
            "pred2points_method": "peaks",
            "output_path": OUTPUT_PATH,
            "filename": f"sau-pacmap-{MODEL_TYPE}-{SEED}",
            "voxelsize": VOXELSIZE,
            "volumesize_vox": VOLUMESIZE_VOX,
            "bordersize_um": BORDERSIZE_UM,
            "threshold": THRESHOLD_D,
            "intensity_as_spacing": True,
            "top_down": False
        }
    })