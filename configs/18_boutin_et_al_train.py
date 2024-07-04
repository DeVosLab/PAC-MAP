from pacmap.utils import crop_stack

dataset_name = "18_boutin_et_al"

# Raw data info
config = {
    "raw": {
        "input_path": f"{dataset_name}/train/raw",
        "voxelsize": [5, 0.64, 0.64]
    }
}

# Preprocessing
config.update({
    "preprocess": {
        "input_path": f"{dataset_name}/train/raw",
        "output_path": f"{dataset_name}/train/preprocessed",
        "channels2store": [0],
        "channels2normalize": [0],
        "current_voxelsize": config['raw']['voxelsize'], 
        "target_voxelsize": config['raw']['voxelsize'],
        "pmins": [0.1],
        "pmaxs": [99.9],
        'crop_func': crop_stack,
        "crop_channel": 0,
        "min_size": 0.01 # As fraction of volume size
    }
})

config.update({
    "points2prob": {
        "input_path": f"{dataset_name}/train/points_csv",
        "output_path": f"{dataset_name}/train",
        "img_path": f"{dataset_name}/train/preprocessed",
        "radi_um": [15, 15, 15],
        "voxelsize": config['raw']['voxelsize'],
        "intensity_as_spacing": True,
        "method": "csv",
        "check_batches": False,
        "save_targets": True # Stored in output_path/targets
    }
})

# Patchify preprocessed images
PATCH_SIZE = [46, 256, 256]
PATCH_STRIDE = [40, 112, 112]
CHANNELS2STORE = [0]
config.update({
    "patch_creation": {
        "preprocessed": {
            "input_path": f"{dataset_name}/train/preprocessed",
            "output_path": f"{dataset_name}/train/patches",
            "patch_size": PATCH_SIZE,
            "patch_stride": PATCH_STRIDE,
            "channels2store": CHANNELS2STORE,
        },
        "targets": {
            "input_path": f"{dataset_name}/train/targets",
            "output_path": f"{dataset_name}/train/targets_patches",
            "patch_size": PATCH_SIZE,
            "patch_stride": PATCH_STRIDE,
            "channels2store": CHANNELS2STORE,
        }
    }
})

# Train nuclei centroid prediction model
config["train"] = {}
seeds = [0, 1, 2]
for SEED in seeds:
    config["train"].update({
        f"scratch-{SEED}": {
            "input_path": f"{dataset_name}/train/patches",
            "binary_path": None,
            "target_path": f"{dataset_name}/train/targets_patches",
            "output_path": f"{dataset_name}/train/models/scratch-{SEED}",
            "split": [0.8, 0.2, 0.0], # Use all data for training and validation
            "random_seed": SEED, # Can be a list of random seeds
            "final_sigmoid": False,
            "normalize_targets": False,
            "min_percentage": None,
            "batch_size": 12,
            "num_epochs": 150,
            "lr": 1e-4,
            "pretrained": None,
            "loss": "MSE",
            "augment_rescale_p": 0.5,
            "augment_rescale_anisotropic": False,
            "f_maps": 16,
            "gpu_id": 3,
            'check_batches': True,
        }
    })