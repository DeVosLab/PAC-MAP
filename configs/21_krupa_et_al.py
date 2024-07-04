dataset_name = "21_krupa_et_al"
config = {}

# Create targets for training
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

# Patchify images
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

INPUT_PATH = f"{dataset_name}/train/patches"
SPLIT = [0.75, 0.25, 0.0]
BATCH_SIZE = 12
NUM_EPOCHS = 300
LR = 1e-4
MODEL_ARCH = "UNet3D"
LOSS = "MSE"
PATIENCE = 10
F_MAPS = 32
DEPTH = 5
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


# Predict with model
INPUT_PATH = f"{dataset_name}/test/imgs"
OUTPUT_PATH_BASE = f"{dataset_name}/test/model_pred"
MODEL_ARCH = "UNet3D"
F_MAPS = 32
DEPTH = 5
MIN_DISTANCE = 5
THRESHOLD_CMAP = 0.1
THRESHOLD_PACMAP = 0.5
THRESHOLD_SEG = 0.2
VOXELSIZE = config['raw']['voxelsize']
MERGE_CLOSE_POINTS = False
DO_BORDER = True
SAVE_CSV = True
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
            "intensity_as_spacing": False,
            "top_down": True,
            "threshold_abs": THRESHOLD_CMAP,
            "gpu_id": GPU_ID,
            "model_type": MODEL_ARCH,
            "f_maps": F_MAPS,
            "depth": DEPTH,
            "voxelsize": VOXELSIZE,
            "min_distance": MIN_DISTANCE,
            "merge_close_points": MERGE_CLOSE_POINTS,
            "do_border": DO_BORDER,
            "save_csv": SAVE_CSV,
            "save_preds": SAVE_PREDS,
            "check_batches": CHECK_BATCHES
        },
        f"pacmap-{SEED}": {
            "input_path": INPUT_PATH,
            "output_path": f"{OUTPUT_PATH_BASE}/pacmap-{SEED}",
            "model_path": f"{dataset_name}/models/pacmap-{SEED}",
            "final_sigmoid": False,
            "intensity_as_spacing": True,
            "top_down": False,
            "threshold_abs": THRESHOLD_PACMAP,
            "gpu_id": GPU_ID,
            "model_type": MODEL_ARCH,
            "f_maps": F_MAPS,
            "depth": DEPTH,
            "voxelsize": VOXELSIZE,
            "min_distance": MIN_DISTANCE,
            "merge_close_points": MERGE_CLOSE_POINTS,
            "do_border": DO_BORDER,
            "save_csv": SAVE_CSV,
            "save_preds": SAVE_PREDS,
            "check_batches": CHECK_BATCHES
        },
        f"seg-{SEED}": {
            "input_path": INPUT_PATH,
            "output_path": f"{OUTPUT_PATH_BASE}/seg-{SEED}",
            "model_path": f"{dataset_name}/models/seg-{SEED}",
            "final_sigmoid": True,
            "threshold_abs": THRESHOLD_SEG,
            "gpu_id": GPU_ID,
            "model_type": MODEL_ARCH,
            "f_maps": F_MAPS,
            "voxelsize": VOXELSIZE,
            "min_distance": MIN_DISTANCE,
            "save_csv": False,
            "save_preds": SAVE_PREDS,
            "check_batches": CHECK_BATCHES
        },
    })

# Performance evaluation
TRUE_PATH = f"{dataset_name}/test/masks"
TRUE2POINTS_METHOD = "masks"
OUTPUT_PATH = f"{dataset_name}/test/model_pred"
SCORE_METHOD = "krupa"
VOLUMESIZE_VOX = [46, 256, 256]
BORDERSIZE_UM = 5
THRESHOLD_SAU = 0.1

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

seeds = [0, 1, 2]
config["performance"] = {}
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
            "threshold": THRESHOLD_SAU,
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
            "threshold": THRESHOLD_CMAP
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
            "threshold": THRESHOLD_PACMAP,
            "intensity_as_spacing": True,
            "top_down": False
        }
    })