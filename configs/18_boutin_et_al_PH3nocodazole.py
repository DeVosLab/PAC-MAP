from pacmap.utils import crop_stack

dataset_name = "18_boutin_et_al"

# Raw data info
config = {
    "raw": {
        "input_path": f"{dataset_name}/PH3nocodazole/raw",
        "voxelsize": [5, 0.64, 0.64]
    }
}

# Preprocessing
config.update({
    "preprocess": {
        "input_path": f"{dataset_name}/PH3nocodazole/raw",
        "output_path": f"{dataset_name}/PH3nocodazole/preprocessed",
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

# Patchify preprocessed images
config.update({
    "patch_creation": {
        "preprocessed": {
            "input_path": f"{dataset_name}/PH3nocodazole/preprocessed",
            "output_path": f"{dataset_name}/PH3nocodazole/patches",
            "patch_size": [46, 256, 256],
            "patch_stride": [40, 224, 224],
            "channels2store": [0],
        }
    }
})

# Predict with model
config.update({
    "pred_model": {
        "scratch": {
            "input_path": f"{dataset_name}/PH3nocodazole/preprocessed",
            "output_path": f"{dataset_name}/PH3nocodazole/model_pred",
            "model_path": f"{dataset_name}/train/models/scratch-0",
            "f_maps": 16,
            "gpu_id": 5,
            "channel2use": 0,
            "min_distance": 7.5,
            "do_border": True,
            "threshold_abs": 0.25,
            "voxelsize": config['raw']['voxelsize'],
            "save_csv": True,
            "save_preds": True,
            "check_batches": False
        } 
    }  
})