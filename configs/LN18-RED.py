from pacmap.utils import crop_stack

dataset_name = "LN18-RED"

# Raw data info
config = {
    "raw": {
        "input_path": f"{dataset_name}/pretrain/raw",
        "voxelsize": [1.9999, 0.3594, 0.3594]
    }
}

# Preprocessing (pacmap/preprocess.py)
config.update({
    "preprocess": {
        "input_path": f"{dataset_name}/pretrain/raw",
        "output_path": f"{dataset_name}/pretrain/preprocessed",
        "output_path_binary": f"{dataset_name}/binary",
        "channels2store": [1],
        "current_voxelsize": config['raw']['voxelsize'], 
        "target_voxelsize": config['raw']['voxelsize'],
        'crop_func': crop_stack,
        "crop_channel": 0,
        "min_size": 0.03 # As fraction of volume size
    }
})

# Binarization (pacmap/binarize.py)
config.update({
    "binarize": {
        "input_path": f"{dataset_name}/pretrain/preprocessed",
        "output_path": f"{dataset_name}/pretrain/binary",
        "channel2use": 0
    },
})

# Patchify preprocessed images (pacmap/patch_creation.py)
config.update({
    "patch_creation": {
        "preprocessed": {
            "input_path": f"{dataset_name}/pretrain/preprocessed",
            "output_path": f"{dataset_name}/pretrain/patches",
            "patch_size": [46, 256, 256],
            "patch_stride": [40, 224, 224],
            "channels2store": [0],
        }
    }
})

# Patchify binary images (pacmap/patch_creation.py)
config['patch_creation'].update({
    "binarized": {
        "input_path": f"{dataset_name}/pretrain/binary",
        "output_path": f"{dataset_name}/pretrain/patches_binary",
        "patch_size": config['patch_creation']['preprocessed']['patch_size'],
        "patch_stride": config['patch_creation']['preprocessed']['patch_stride'],
        "channels2store": [0]
    }
})

# Generate weak targets (pacmap/weak_targets.py)
config.update({
    "weak_targets": {
        "input_path_grayscale": f"{dataset_name}/pretrain/patches",
        "input_path_binary": f"{dataset_name}/pretrain/patches_binary",
        "output_path": f"{dataset_name}/pretrain/weak_targets_cmap",
        "radi_um": [10, 10, 10],
        "voxelsize": config['raw']['voxelsize'],
        "min_distance": 5,
        "exclude_border": True,
        "save_points_csv": True,
        "save_targets": True
    }
})

# Generate weak targets with proximity adjustment, using the same points as 
# the original weak targets (pacmap/weak_targets.py)
config.update({
    "points2prob": {
        "weak_targets_d": {
            "input_path": f"{dataset_name}/pretrain/weak_targets_cmap/points_csv",
            "output_path": f"{dataset_name}/pretrain/weak_targets_pacmap",
            "img_path": f"{dataset_name}/pretrain/patches",
            "radi_um": [10, 10, 10],
            "intensity_as_spacing": True,
            "voxelsize": config['raw']['voxelsize'],
            "method": "csv",
            "check_batches": True,
            "save_targets": True # Stored in output_path/targets
        },
    }
})

# Train nuclei centroid prediction models (pacmap/train.py)
INPUT_PATH = f"{dataset_name}/pretrain/patches"
INPUT_PATH_BINARY = f"{dataset_name}/pretrain/patches_binary"
TARGET_PATH_BASE = f"{dataset_name}/pretrain/weak_targets"
OUTPUT_PATH_BASE = f"{dataset_name}/pretrain/models"

SPLIT = [0.8, 0.2, 0.0]
BATCH_SIZE = 4
NUM_EPOCHS_PRE = 100
NUM_EPOCHS = 150
LR = 1e-4
MODEL_ARCH = "UNet3D"
LOSS = "MSE"
PATIENCE = 10
F_MAPS = 32
DEPTH = 4
CHECK_BATCHES = True
MIN_PERCENTAGE = 1/3
AUGMENT_RESCALE_P = 0.0
GPU_ID = 0

seeds = [0, 1, 2]
config['train'] = {}
for SEED in seeds:
    config['train'].update({
        f"cmap-pretrain-{SEED}": {
            "input_path": INPUT_PATH,
            "binary_path": INPUT_PATH_BINARY,
            "target_path": f"{TARGET_PATH_BASE}_cmap/targets_pac",
            "output_path": f"{OUTPUT_PATH_BASE}/cmap-pretrain-{SEED}",
            "gpu_id": GPU_ID,
            "random_seed": SEED,
            "final_sigmoid": True,
            "normalize_targets": True,
            "model_type": MODEL_ARCH,
            "loss": LOSS,
            "patience": PATIENCE,
            "split": SPLIT,
            "batch_size": BATCH_SIZE,
            "num_epochs": NUM_EPOCHS_PRE,
            "lr": LR,
            "min_percentage": MIN_PERCENTAGE,
            "augment_rescale_p": AUGMENT_RESCALE_P,
            "f_maps": F_MAPS,
            "depth": DEPTH,
            "check_batches": CHECK_BATCHES,
        },
        f"pacmap-pretrain-{SEED}": {
            "input_path": INPUT_PATH,
            "binary_path": INPUT_PATH_BINARY,
            "target_path": f"{TARGET_PATH_BASE}_pacmap/targets_pac",
            "output_path": f"{OUTPUT_PATH_BASE}/pacmap-pretrain-{SEED}",
            "gpu_id": GPU_ID,
            "random_seed": SEED,
            "final_sigmoid": False,
            "normalize_targets": False,
            "model_type": MODEL_ARCH,
            "loss": LOSS,
            "patience": PATIENCE,
            "split": SPLIT,
            "batch_size": BATCH_SIZE,
            "num_epochs": NUM_EPOCHS_PRE,
            "lr": LR,
            "min_percentage": MIN_PERCENTAGE,
            "augment_rescale_p": AUGMENT_RESCALE_P,
            "f_maps": F_MAPS,
            "depth": DEPTH,
            "check_batches": CHECK_BATCHES,
        },
    })