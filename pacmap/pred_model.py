from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm
import tifffile
import ast
import numpy as np
import pandas as pd
import torch

import importlib
pytorch3dunet = importlib.import_module('..pytorch-3dunet.pytorch3dunet.unet3d.model')

from .utils import squeeze_to_ndim, unsqueeze_to_ndim, merge_close_points
from .train import get_data, CentroidDataset, create_model
from .prob2points import get_points


def get_model_file(path, pattern='*_best_model.pt'):
    ''' Get model file from path. If path is a directory, look for the best model file in the directory.

    Parameters
    ----------
    path : str or Path
        Path to the model file or directory containing the model file
    pattern : str, optional
        Pattern to look for the model file in the directory, by default '*_best_model.pt'

    Returns
    -------
    Path
        Path to the model file
    '''

    if path.suffix == '.pt':
        return path
    elif path.is_dir():
        path = Path(path) if not isinstance(path, Path) else path
        files = list(path.glob(pattern))
        if len(files) == 0:
            raise FileNotFoundError(f'No .pt files found in {path}')
        elif len(files) > 1:
            raise ValueError(f'More than one .pt file found in {path}')
        return files[0]
    else:
        raise FileNotFoundError(f'No .pt files found in {path}')

def main(**kwargs):
    input_path = Path(kwargs['input_path'])
    df = get_data(input_path, check_batches=kwargs['check_batches'])
    dataset = CentroidDataset(df, transform=None)
    n_patches = len(dataset)
    
    # Get metadata from the images 
    with tifffile.TiffFile(dataset[0]['input_filename']) as tif:
        if tif.imagej_metadata:
            try:
                metadata = tif.imagej_metadata
                metadata['patch_size'] = list(ast.literal_eval(metadata['patch_size']))
                metadata['patch_stride'] = list(ast.literal_eval(metadata['patch_stride']))
                metadata['patches_structure'] = list(ast.literal_eval(metadata['patches_structure']))
                metadata['original_img_shape'] = list(ast.literal_eval(metadata['original_img_shape']))
                
                # Remove/overwrite channel axis from metadata
                metadata['patch_size'].pop(1)
                metadata['patch_stride'].pop(1) 
                metadata['patches_structure'].pop(1)
                metadata['original_img_shape'].pop(1)
            except:
                metadata = {}      
        else:
            metadata = {}
        metadata['channels'] = 1
        metadata['axes'] = 'ZYX'
        
    
    output_path = Path(kwargs['output_path'])
    output_path.mkdir(exist_ok=True, parents=True)

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{kwargs['gpu_id']}")
    elif torch.backends.mps.is_available() and kwargs['mps']:
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    # Load model
    model = create_model(
        model_type=kwargs['model_type'],
        in_channels=1,
        out_channels=1,
        final_sigmoid=kwargs['final_sigmoid'],
        f_maps=kwargs['f_maps'],
        depth=kwargs['depth']
    )
    model.to(device)
    model_path = get_model_file(kwargs['model_path'])
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # Predict on each patch
    for i in tqdm(range(n_patches)):
        sample = dataset[i]
        patch = torch.squeeze(sample['input'])
        if patch.ndim == 4:
            patch = sample['input'][:, kwargs['channel2use'], ...]
        eps = 1e-20
        patch = (patch - patch.min()) / (patch.max() - patch.min() + eps)
        patch = unsqueeze_to_ndim(patch, 5).to(device)
        print('\nPredicting centroid probabilities')
        # Predict in half precision
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            with torch.no_grad():
                output = model(patch)
        output = squeeze_to_ndim(output, 3).cpu().numpy().astype(float)

        if kwargs['save_npy'] or kwargs['save_csv'] or kwargs['save_points_imgs']:
            print('Finding peaks')
            points = get_points(
                output,
                min_distance = kwargs['min_distance'],
                threshold_abs = kwargs['threshold_abs'],
                exclude_border= not kwargs['do_border'],
                intensity_as_spacing = kwargs['intensity_as_spacing'],
                top_down = kwargs['top_down'],
                voxelsize = kwargs['voxelsize']
                )

            if kwargs['merge_close_points']:
                points = merge_close_points(points, kwargs['voxelsize'], kwargs['min_distance'])
            points = points.astype(int)
        
        print('Saving predictions')
        filename = Path(sample['input_filename'])
        batch = Path(sample['input_filename']).parent.name
        filename = filename.stem

        if kwargs['save_preds']:
            filepath = output_path.joinpath('preds', batch, filename + '.tif')
            filepath.parent.mkdir(exist_ok=True, parents=True)
            tifffile.imwrite(
                filepath,
                output.astype(np.float32),
                imagej=True,
                metadata=metadata,
                compression='zlib'
            )
        if kwargs['save_npy']:
            filepath = output_path.joinpath('points_npy', batch, filename + '.npy')
            filepath.parent.mkdir(exist_ok=True, parents=True)
            np.save(filepath, points)

        if kwargs['save_csv']:
            filepath = output_path.joinpath('points_csv', batch, filename + '.csv')
            filepath.parent.mkdir(exist_ok=True, parents=True)
            points = pd.DataFrame(points)
            if len(points.columns) == 3: # do not put header in empty predictions
                points.columns = ['axis-0', 'axis-1', 'axis-2']
            points.to_csv(filepath, index=True, header=True)

        if kwargs['save_points_imgs']:
            filepath = output_path.joinpath('points_imgs', batch, filename + '.tif')
            filepath.parent.mkdir(exist_ok=True, parents=True)
            points_img = np.zeros(output.shape, dtype=bool)
            points_img[tuple(points.T)] = True
            tifffile.imwrite(
                filepath,
                points_img.astype(np.float32),
                imagej=True,
                metadata=metadata,
                compression='zlib'
            )


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('-i','--input_path', type=str, default='../data/Spheroids_LSFM/patches_manual_annotated/data',
        help='Define input path with image patches')
    parser.add_argument('-o','--output_path', type=str, default='../data/Spheroids_LSFM/patches_manual_annotated/model_predictions',
        help='Define path where predictions on image patches must be saved')
    parser.add_argument('--check_batches', action='store_true',
                        help='Look for images in batch subfolders')
    parser.add_argument('--file_extension', type=str, default='.tif',
        help='Define the file extension of the input files')
    parser.add_argument('--channel2use', nargs=1, type=int, default=0,
        help='Define which channels of the input images should be used')
    parser.add_argument('--model_path', type=str, default=None,
        help='Define the path of the trained model to be used')
    parser.add_argument('--model_type', type=str, choices=('UNet3D', 'ResidualUNet3D') , default='UNet3D',
        help='Model type (default: UNet3D)')
    parser.add_argument('--final_sigmoid', action='store_true', default=False,
        help='Whether to use a sigmoid activation function in the final layer (default: True)')
    parser.add_argument('--f_maps', type=int, default=16,
        help='Number of feature maps in the first layer of the network')
    parser.add_argument('--depth', type=int, default=4,
        help='Number of encoder layers in the network')
    parser.add_argument('--min_distance', type=int, default=1,
        help='Define minimal distance allowed for separating peaks')
    parser.add_argument('--do_border', action='store_true',
        help='Include peaks at the border of the image')
    parser.add_argument('--intensity_as_spacing', action='store_true',
        help='Use intensity values as spacing for peak detection')
    parser.add_argument('--top_down', action='store_true',
        help='Use top-down approach for peak detection (higher intensity first)')
    parser.add_argument('--voxelsize', nargs=3, type=float, default=[1.9999, 0.3594, 0.3594],
        help='Define the voxelsize in um for z, y, x')
    parser.add_argument('--threshold_abs', type=float, default=None,
        help='Define minimum intensity for peaks')
    parser.add_argument('--merge_close_points', type=bool, default=True,
        help='Merge points that are too close')
    parser.add_argument('--gpu_id', type=int, default=0, 
        help='GPU id to use (default: 0)')
    parser.add_argument('--mps', action='store_true',
        help='Use Metal Performance Shaders (macos) to speed up prediction')
    parser.add_argument('--save_npy', action='store_true', 
        help='Store points as .npy')
    parser.add_argument('--save_csv', action='store_true', 
        help='Store points as .csv')   
    parser.add_argument('--save_points_imgs', action='store_true', 
        help='Store points_imgs as .tif')
    parser.add_argument('--save_preds', action='store_true', 
        help='Store predictions as .tif')
    args, _ = parser.parse_known_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(**vars(args))