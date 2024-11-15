import numpy as np

dataset_name = "21_krupa_et_al"
config = {}

# CREATE TARGETS FOR TRAINING
INPUT_PATH = f"{dataset_name}/train/masks"
OUTPUT_PATH_BASE = f"{dataset_name}/train"
IMG_PATH = f"{dataset_name}/train/imgs"
RADI_UM = 5
VOXELSIZE = [2.5, 0.75, 0.75]
METHOD = "masks"
CHECK_BATCHES = False
SAVE_TARGETS = True
config.update({
    "points2prob": {
        "targets_cmap":{
            "input_path": INPUT_PATH,
            "output_path": f"{OUTPUT_PATH_BASE}/targets_cmap",
            "img_path": IMG_PATH,
            "radi_um": RADI_UM,
            "intensity_as_spacing": False,
            "voxelsize": VOXELSIZE,
            "method": METHOD,
            "check_batches": CHECK_BATCHES,
            "save_targets": SAVE_TARGETS # Stored in output_path/targets
        },
        "targets_pacmap":{
            "input_path": INPUT_PATH,
            "output_path": f"{OUTPUT_PATH_BASE}/targets_pacmap",
            "img_path": IMG_PATH,
            "radi_um": RADI_UM,
            "intensity_as_spacing": True,
            "voxelsize": VOXELSIZE,
            "method": METHOD,
            "check_batches": CHECK_BATCHES,
            "save_targets": SAVE_TARGETS # Stored in output_path/targets
        }
    }
})

# PATCHIFY IMAGES
PATCH_SIZE = [32, 112, 112]
PATCH_STRIDE = [32, 112, 112]
CHANNELS2STORE = [0]
STORE_BATCHES = False
config.update({
    "patch_creation": {
        "imgs": {
            "input_path": IMG_PATH,
            "output_path": f"{dataset_name}/train/patches",
            "patch_size": PATCH_SIZE,
            "patch_stride": PATCH_STRIDE,
            "channels2store": CHANNELS2STORE,
            "store_batches": STORE_BATCHES
        },
        "targets_cmap": {
            "input_path": f"{dataset_name}/train/targets_cmap/targets",
            "output_path": f"{dataset_name}/train/targets_cmap/targets_patches",
            "patch_size": PATCH_SIZE,
            "patch_stride": PATCH_STRIDE,
            "channels2store": CHANNELS2STORE,
            "store_batches": STORE_BATCHES
        },
        "targets_pacmap": {
            "input_path": f"{dataset_name}/train/targets_pacmap/targets",
            "output_path": f"{dataset_name}/train/targets_pacmap/targets_patches",
            "patch_size": PATCH_SIZE,
            "patch_stride": PATCH_STRIDE,
            "channels2store": CHANNELS2STORE,
            "store_batches": STORE_BATCHES
        },
        "masks": {
            "input_path": f"{dataset_name}/train/masks",
            "output_path": f"{dataset_name}/train/masks/masks_patches",
            "patch_size": PATCH_SIZE,
            "patch_stride": PATCH_STRIDE,
            "channels2store": CHANNELS2STORE,
            "store_batches": STORE_BATCHES
        }
    }
})

# TRAIN STANDARD 3D U-NET MODELS
INPUT_PATH = f"{dataset_name}/train/patches"
SPLIT = [0.75, 0.25, 0.0]
BATCH_SIZE = 12
NUM_EPOCHS = 300
LR = 1e-4
MODEL_ARCH = "UNet3D"
LOSS = "MSE"
PATIENCE = 10
F_MAPS = 32
DEPTH = 4
RESCALE_P = 0.0
CHECK_BATCHES = False
GPU_ID = 0

seeds = [0, 1, 2]
config['train'] = {}
for SEED in seeds:
    config['train'].update({
        f"cmap-{SEED}": {
            "input_path": INPUT_PATH,
            "target_path": f"{dataset_name}/train/targets_cmap/targets_patches",
            "output_path": f"{dataset_name}/models/cmap-{SEED}",
            "random_seed": SEED,
            "gpu_id": GPU_ID,
            "final_sigmoid": True,
            "normalize_targets": True,
            "split": SPLIT,
            "batch_size": BATCH_SIZE,
            "num_epochs": NUM_EPOCHS,
            "lr": LR,
            "model_type": MODEL_ARCH,
            "depth": DEPTH,
            "loss": LOSS,
            "patience": PATIENCE,
            "augment_rescale_p": RESCALE_P,
            "f_maps": F_MAPS,
            'check_batches': CHECK_BATCHES,
        },
        f"pacmap-{SEED}": {
            "input_path": INPUT_PATH,
            "target_path": f"{dataset_name}/train/targets_pacmap/targets_patches",
            "output_path": f"{dataset_name}/models/pacmap-{SEED}",
            "random_seed": SEED,
            "gpu_id": GPU_ID,
            "final_sigmoid": False,
            "normalize_targets": False,
            "split": SPLIT,
            "batch_size": BATCH_SIZE,
            "num_epochs": NUM_EPOCHS,
            "lr": LR,
            "model_type": MODEL_ARCH,
            "depth": DEPTH,
            "loss": LOSS,
            "patience": PATIENCE,
            "augment_rescale_p": RESCALE_P,
            "f_maps": F_MAPS,
            'check_batches': CHECK_BATCHES,
        },
        f"seg-{SEED}": {
            "input_path": INPUT_PATH,
            "target_path": f"{dataset_name}/train/masks/masks_patches",
            "output_path": f"{dataset_name}/models/seg-{SEED}",
            "random_seed": SEED,
            "gpu_id": GPU_ID,
            "final_sigmoid": True,
            "normalize_targets": False,
            "split": SPLIT,
            "batch_size": BATCH_SIZE,
            "num_epochs": NUM_EPOCHS,
            "lr": LR,
            "model_type": MODEL_ARCH,
            "depth": DEPTH,
            "loss": "TverskyLoss",
            "patience": PATIENCE,
            "augment_rescale_p": RESCALE_P,
            "f_maps": F_MAPS,
            'check_batches': CHECK_BATCHES,
        }
    })

# TRAIN SAU-NET (Guo et al., 2022) BASED MODELS
INPUT_PATH = f"{dataset_name}/train/patches"
SPLIT = [0.75, 0.25, 0.0]
BATCH_SIZE = 12
NUM_EPOCHS = 300
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
RESCALE_P = 0.0
CHECK_BATCHES = False
GPU_ID = 0

seeds = [0, 1, 2]
for SEED in seeds:
    config['train'].update({
        f"sau-cmap-{SEED}": {
            "input_path": INPUT_PATH,
            "target_path": f"{dataset_name}/train/targets_cmap/targets_patches",
            "output_path": f"{dataset_name}/models/sau-cmap-{SEED}",
            "random_seed": SEED,
            "gpu_id": GPU_ID,
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
            "depth": DEPTH,
            "loss": LOSS,
            "augment_rescale_p": RESCALE_P,
            "f_maps": F_MAPS,
            'check_batches': CHECK_BATCHES,
        },
        f"sau-pacmap-{SEED}": {
            "input_path": INPUT_PATH,
            "target_path": f"{dataset_name}/train/targets_pacmap/targets_patches",
            "output_path": f"{dataset_name}/models/sau-pacmap-{SEED}",
            "random_seed": SEED,
            "gpu_id": GPU_ID,
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
            "depth": DEPTH,
            "loss": LOSS,
            "augment_rescale_p": RESCALE_P,
            "f_maps": F_MAPS,
            'check_batches': CHECK_BATCHES,
        },
        f"sau-seg-{SEED}": {
            "input_path": INPUT_PATH,
            "target_path": f"{dataset_name}/train/masks/masks_patches",
            "output_path": f"{dataset_name}/models/sau-seg-{SEED}",
            "random_seed": SEED,
            "gpu_id": GPU_ID,
            "final_sigmoid": True,
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
            "depth": DEPTH,
            "loss": "TverskyLoss",
            "augment_rescale_p": RESCALE_P,
            "f_maps": F_MAPS,
            'check_batches': CHECK_BATCHES,
        }
    })

# PREDICT WITH STANDARD 3D U-NET MODELS
INPUT_PATH = f"{dataset_name}/test/imgs"
OUTPUT_PATH_BASE = f"{dataset_name}/test/model_pred"
MODEL_ARCH = "UNet3D"
F_MAPS = 32
DEPTH = 4
VOXELSIZE = config['raw']['voxelsize']
SAVE_CSV = False
SAVE_PREDS = True
CHECK_BATCHES = False
GPU_ID = 0

seeds = [0, 1, 2]
config["pred_model"] = {}
for SEED in seeds:
    config["pred_model"].update({
        f"cmap-{SEED}": {
            "input_path": INPUT_PATH,
            "output_path": f"{OUTPUT_PATH_BASE}/cmap-{SEED}",
            "model_path": f"{dataset_name}/models/cmap-{SEED}",
            "final_sigmoid": True,
            "gpu_id": GPU_ID,
            "model_type": MODEL_ARCH,
            "f_maps": F_MAPS,
            "depth": DEPTH,
            "voxelsize": VOXELSIZE,
            "save_csv": SAVE_CSV,
            "save_preds": SAVE_PREDS,
            "check_batches": CHECK_BATCHES
        },
        f"pacmap-{SEED}": {
            "input_path": INPUT_PATH,
            "output_path": f"{OUTPUT_PATH_BASE}/pacmap-{SEED}",
            "model_path": f"{dataset_name}/models/pacmap-{SEED}",
            "final_sigmoid": False,
            "gpu_id": GPU_ID,
            "model_type": MODEL_ARCH,
            "f_maps": F_MAPS,
            "depth": DEPTH,
            "voxelsize": VOXELSIZE,
            "save_csv": SAVE_CSV,
            "save_preds": SAVE_PREDS,
            "check_batches": CHECK_BATCHES
        },
        f"seg-{SEED}": {
            "input_path": INPUT_PATH,
            "output_path": f"{OUTPUT_PATH_BASE}/seg-{SEED}",
            "model_path": f"{dataset_name}/models/seg-{SEED}",
            "final_sigmoid": True,
            "gpu_id": GPU_ID,
            "model_type": MODEL_ARCH,
            "f_maps": F_MAPS,
            "depth": DEPTH,
            "voxelsize": VOXELSIZE,
            "save_csv": SAVE_CSV,
            "save_preds": SAVE_PREDS,
            "check_batches": CHECK_BATCHES
        },
    })

# PREDICT WITH SAU-NET (Guo et al., 2022) BASED MODELS
INPUT_PATH = f"{dataset_name}/test/imgs"
OUTPUT_PATH_BASE = f"{dataset_name}/test/model_pred"
MODEL_ARCH = "SAUNet3D"
F_MAPS = 32
DEPTH = 4
VOXELSIZE = config['raw']['voxelsize']
SAVE_CSV = False
SAVE_PREDS = True
CHECK_BATCHES = False
GPU_ID = 0

seeds = [0, 1, 2]
for SEED in seeds:
    config["pred_model"].update({
        f"sau-cmap-{SEED}": {
            "input_path": INPUT_PATH,
            "output_path": f"{OUTPUT_PATH_BASE}/sau-cmap-{SEED}",
            "model_path": f"{dataset_name}/models/sau-cmap-{SEED}",
            "final_sigmoid": True,
            "gpu_id": GPU_ID,
            "model_type": MODEL_ARCH,
            "f_maps": F_MAPS,
            "depth": DEPTH,
            "voxelsize": VOXELSIZE,
            "save_csv": SAVE_CSV,
            "save_preds": SAVE_PREDS,
            "check_batches": CHECK_BATCHES
        },
        f"sau-pacmap-{SEED}": {
            "input_path": INPUT_PATH,
            "output_path": f"{OUTPUT_PATH_BASE}/sau-pacmap-{SEED}",
            "model_path": f"{dataset_name}/models/sau-pacmap-{SEED}",
            "final_sigmoid": False,
            "gpu_id": GPU_ID,
            "model_type": MODEL_ARCH,
            "f_maps": F_MAPS,
            "depth": DEPTH,
            "voxelsize": VOXELSIZE,
            "save_csv": SAVE_CSV,
            "save_preds": SAVE_PREDS,
            "check_batches": CHECK_BATCHES
        },
        f"sau-seg-{SEED}": {
            "input_path": INPUT_PATH,
            "output_path": f"{OUTPUT_PATH_BASE}/sau-seg-{SEED}",
            "model_path": f"{dataset_name}/models/sau-seg-{SEED}",
            "final_sigmoid": True,
            "gpu_id": GPU_ID,
            "model_type": MODEL_ARCH,
            "f_maps": F_MAPS,
            "voxelsize": VOXELSIZE,
            "save_csv": SAVE_CSV,
            "save_preds": SAVE_PREDS,
            "check_batches": CHECK_BATCHES
        },
    })

# PERFORMANCE EVALUATION
TRUE_PATH = f"{dataset_name}/test/masks"
TRUE2POINTS_METHOD = "masks"
OUTPUT_PATH = f"{dataset_name}/test/model_pred"
SCORE_METHOD = "krupa"
VOLUMESIZE_VOX = [46, 256, 256]
BORDERSIZE_UM = 3 # inline with Krupa et al. (2021)
n_steps = 15 # for thresholds
t_min = 0
t_max = 1
THRESHOLD_SEG = np.linspace(t_min, t_max, n_steps).tolist() # for segmentation based methods
THRESHOLD = np.linspace(t_min, t_max, n_steps).tolist() # for probability based methods
THRESHOLD_D = (12*np.linspace(t_min, t_max, n_steps)).tolist()  # for proximity adjusted probability based methods
                                                                # ranging between 0 and median nearest neighbour distance
MIN_DISTANCE = 5

## Stardist
config.update({
    "performance": {
        "stardist": {
            "true_path": TRUE_PATH,
            "true2points_method": TRUE2POINTS_METHOD,
            "pred_path": f"{dataset_name}/test/model_pred/stardist",
            "pred2points_method": "masks",
            "output_path": OUTPUT_PATH,
            "filename": "stardist",
            "score_method": SCORE_METHOD,
            "voxelsize": VOXELSIZE,
            "volumesize_vox": VOLUMESIZE_VOX,
            "bordersize_um": BORDERSIZE_UM
        }
    }
})

## Standard 3D U-Net and SAU-Net (Guo et al., 2022) based models
seeds = [0, 1, 2]
for SEED in seeds:
    config["performance"].update({
        f"numorph-{SEED}": {
            "true_path": TRUE_PATH,
            "true2points_method": TRUE2POINTS_METHOD,
            "pred_path": f"{dataset_name}/test/model_pred/seg-{SEED}/preds",
            "pred2points_method": "cc",
            "output_path": OUTPUT_PATH,
            "filename": f"numorph-{SEED}",
            "score_method": SCORE_METHOD,
            "voxelsize": VOXELSIZE,
            "volumesize_vox": VOLUMESIZE_VOX,
            "bordersize_um": BORDERSIZE_UM,
            "threshold": THRESHOLD_SEG
        },
        f"sau-net-{SEED}": {
            "true_path": TRUE_PATH,
            "true2points_method": TRUE2POINTS_METHOD,
            "pred_path": f"{dataset_name}/test/model_pred/cmap-{SEED}/preds",
            "pred2points_method": "cc",
            "output_path": OUTPUT_PATH,
            "filename": f"sau-net-{SEED}",
            "score_method": SCORE_METHOD,
            "voxelsize": VOXELSIZE,
            "volumesize_vox": VOLUMESIZE_VOX,
            "bordersize_um": BORDERSIZE_UM,
            "threshold": THRESHOLD,
        },
        f"cmap-{SEED}": {
            "true_path": TRUE_PATH,
            "true2points_method": TRUE2POINTS_METHOD,
            "pred_path": f"{dataset_name}/test/model_pred/cmap-{SEED}/preds",
            "pred2points_method": "peaks",
            "output_path": OUTPUT_PATH,
            "filename": f"cmap-{SEED}",
            "score_method": SCORE_METHOD,
            "voxelsize": VOXELSIZE,
            "volumesize_vox": VOLUMESIZE_VOX,
            "bordersize_um": BORDERSIZE_UM,
            "threshold": THRESHOLD
        },
        f"pacmap-{SEED}": {
            "true_path": TRUE_PATH,
            "true2points_method": TRUE2POINTS_METHOD,
            "pred_path": f"{dataset_name}/test/model_pred/pacmap-{SEED}/preds",
            "pred2points_method": "peaks",
            "output_path": OUTPUT_PATH,
            "filename": f"pacmap-{SEED}",
            "score_method": SCORE_METHOD,
            "voxelsize": VOXELSIZE,
            "volumesize_vox": VOLUMESIZE_VOX,
            "bordersize_um": BORDERSIZE_UM,
            "threshold": THRESHOLD_D,
            "intensity_as_spacing": True,
            "top_down": False
        },
        f"sau-numorph-{SEED}": {
            "true_path": TRUE_PATH,
            "true2points_method": TRUE2POINTS_METHOD,
            "pred_path": f"{dataset_name}/test/model_pred/sau-seg-{SEED}/preds",
            "pred2points_method": "cc",
            "output_path": OUTPUT_PATH,
            "filename": f"sau-numorph-{SEED}",
            "score_method": SCORE_METHOD,
            "voxelsize": VOXELSIZE,
            "volumesize_vox": VOLUMESIZE_VOX,
            "bordersize_um": BORDERSIZE_UM,
            "threshold": THRESHOLD_SEG
        },
        f"sau-sau-net-{SEED}": {
            "true_path": TRUE_PATH,
            "true2points_method": TRUE2POINTS_METHOD,
            "pred_path": f"{dataset_name}/test/model_pred/sau-cmap-{SEED}/preds",
            "pred2points_method": "cc",
            "output_path": OUTPUT_PATH,
            "filename": f"sau-sau-net-{SEED}",
            "score_method": SCORE_METHOD,
            "voxelsize": VOXELSIZE,
            "volumesize_vox": VOLUMESIZE_VOX,
            "bordersize_um": BORDERSIZE_UM,
            "threshold": THRESHOLD,
        },
        f"sau-cmap-{SEED}": {
            "true_path": TRUE_PATH,
            "true2points_method": TRUE2POINTS_METHOD,
            "pred_path": f"{dataset_name}/test/model_pred/sau-cmap-{SEED}/preds",
            "pred2points_method": "peaks",
            "output_path": OUTPUT_PATH,
            "filename": f"sau-cmap-{SEED}",
            "score_method": SCORE_METHOD,
            "voxelsize": VOXELSIZE,
            "volumesize_vox": VOLUMESIZE_VOX,
            "bordersize_um": BORDERSIZE_UM,
            "threshold": THRESHOLD
        },
        f"sau-pacmap-{SEED}": {
            "true_path": TRUE_PATH,
            "true2points_method": TRUE2POINTS_METHOD,
            "pred_path": f"{dataset_name}/test/model_pred/sau-pacmap-{SEED}/preds",
            "pred2points_method": "peaks",
            "output_path": OUTPUT_PATH,
            "filename": f"sau-pacmap-{SEED}",
            "score_method": SCORE_METHOD,
            "voxelsize": VOXELSIZE,
            "volumesize_vox": VOLUMESIZE_VOX,
            "bordersize_um": BORDERSIZE_UM,
            "threshold": THRESHOLD_D,
            "intensity_as_spacing": True,
            "top_down": False
        }
    })