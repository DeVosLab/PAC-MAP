dataset_name = "18_boutin_et_al"

# Raw data info
config = {
    "raw": {
        "input_path": f"{dataset_name}/test/raw",
        "voxelsize": [5, 0.64, 0.64]
    }
}

# Preprocessing
config.update({
    "preprocess": {
        "input_path": f"{dataset_name}/test/raw",
        "output_path": f"{dataset_name}/test/preprocessed",
        "channels2store": [0],
        "current_voxelsize": config['raw']['voxelsize'], 
        "target_voxelsize": config['raw']['voxelsize'],
        "pmins": [0.1],
        "pmaxs": [99.9],
    }
})

# Predict with model
INPUT_PATH = f"{dataset_name}/test/preprocessed"
OUTPUT_PATH_BASE = f"{dataset_name}/test/model_pred"
MODEL_PATH_BASE = f"{dataset_name}/train/models"
F_MAPS = 16
GPU_ID = 0
MIN_DISTANCE = 7.5
DO_BORDER = True
THRESHOLD_ABS = 0.25
VOXELSIZE = config['raw']['voxelsize']
SAVE_CSV = True
SAVE_PREDS = True
CHECK_BATCHES = False

config["pred_model"] = {}
seeds = [0, 1, 2]
for SEED in seeds:
    config["pred_model"].update({
        f"scratch-{SEED}": {
            "input_path": f"{dataset_name}/test/preprocessed",
            "output_path": f"{dataset_name}/test/model_pred/scratch-{SEED}",
            "model_path": f"{dataset_name}/train/models/scratch-{SEED}",
            "f_maps": F_MAPS,
            "gpu_id": GPU_ID,
            "min_distance": MIN_DISTANCE,
            "do_border": DO_BORDER,
            "threshold_abs": THRESHOLD_ABS,
            "voxelsize": VOXELSIZE,
            "save_csv": SAVE_CSV,
            "save_preds": SAVE_PREDS,
            "check_batches": CHECK_BATCHES
        },
    })
            