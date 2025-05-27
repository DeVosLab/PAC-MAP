from dotenv import load_dotenv
from importlib.machinery import SourceFileLoader
from pathlib import Path
from tqdm import tqdm
import ast
import numpy as np
import pandas as pd
import tifffile
from skimage.morphology import binary_closing
from scipy.ndimage import binary_fill_holes
import os



import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from pacmap.utils import merge_close_points

def depatchify_merge(patches, shape, patch_stride, eps=1e-6):
    if patches.ndim == 6:
        D, H, W = patches.shape[:3]
        p_size_D, p_size_H, p_size_W = patches.shape[3:]
        step_D, step_H, step_W = patch_stride
    if patches.ndim == 8:
        D, C, H, W = patches.shape[:4]
        p_size_D, p_size_C, p_size_H, p_size_W = patches.shape[4:]
        step_D, step_C, step_H, step_W = patch_stride

    img = np.zeros(shape, dtype=np.float32)
    n_values = np.full(shape, eps, dtype=np.float32)

    if patches.ndim == 6:
        for d in range(D):
                for h in range(H):
                    for w in range(W):
                        img[
                            d*step_D:d*step_D+p_size_D,
                            h*step_H:h*step_H+p_size_H,
                            w*step_W:w*step_W+p_size_W
                        ] += patches[d,h,w,]
                        n_values[
                            d*step_D:d*step_D+p_size_D,
                            h*step_H:h*step_H+p_size_H,
                            w*step_W:w*step_W+p_size_W
                        ] += 1
    elif patches.ndim == 8:
        for d in range(D):
            for c in range(C):
                for h in range(H):
                    for w in range(W):
                        img[
                            d*step_D:d*step_D+p_size_D,
                            c*step_C:d*step_C+p_size_C,
                            h*step_H:h*step_H+p_size_H,
                            w*step_W:w*step_W+p_size_W
                        ] += patches[d,c,h,w,]
                        n_values[
                            d*step_D:d*step_D+p_size_D,
                            c*step_C:d*step_C+p_size_C,
                            h*step_H:h*step_H+p_size_H,
                            w*step_W:w*step_W+p_size_W
                        ] += 1
    img = img / n_values

    return img


def fill_binary_holes_border(binary, footprint=np.ones((1, 3, 3))):
    Z, Y, X = binary.shape
    binary = binary_closing(binary, footprint=footprint)
    for z in range(Z):
        binary[z] = binary_fill_holes(binary[z])
    for x in range(X):
        binary[:, :, x] = binary_fill_holes(binary[:, :, x])
    return binary


def mask_points(points, binary):
    points = np.array(points).astype(int)
    Z, Y, X = binary.shape
    points_img = np.zeros((Z, Y, X), dtype=np.uint8)
    points_img[points[:,0], points[:,1], points[:,2]] = 1
    points_img = np.where(binary, points_img, 0)
    points = np.argwhere(points_img)
    return points


load_dotenv()
data_path = Path(os.getenv('DATA_PATH'))
dataset_name = '241012_CO-MG_jvdd'
model = 'pacmap-finetuned-0'
do_fill_holes = True
do_mask_points = False

# Load dataset configuration stored as dict in python file
config_file = Path(f'configs/{dataset_name}.py')
config = SourceFileLoader(config_file.name, str(config_file)).load_module().config

input_path = data_path.joinpath(f'{dataset_name}/patches')
binary_path = data_path.joinpath(f'{dataset_name}/patches_binary')
preds_path = data_path.joinpath(f'{dataset_name}/model_pred/{model}/preds')
coord_path = data_path.joinpath(f'{dataset_name}/model_pred/{model}/points_csv')
output_path = data_path.joinpath(f'{dataset_name}/depatchified')
output_path.mkdir(exist_ok=True, parents=True)

samples = [f.name for f in input_path.iterdir() if f.is_dir()]
for sample in tqdm(samples, desc='Depatchifying samples', colour='green', leave=True):
    input_path_sample = input_path.joinpath(sample)
    binary_path_sample = binary_path.joinpath(sample)
    preds_path_sample = preds_path.joinpath(sample)
    coord_path_sample = coord_path.joinpath(sample)
    output_path_sample = output_path.joinpath(sample)
    output_path_sample.mkdir(exist_ok=True, parents=True)

    # Depatchify images
    files = sorted([f for f in Path(input_path_sample).iterdir() if f.is_file() and f.suffix == '.tif' and not f.stem.startswith('.')])
    imgs = []
    for i, f in enumerate(tqdm(files, desc='Depatchifying images', colour='blue', leave=True)):
        with tifffile.TiffFile(f) as tif:
            patch = tif.asarray()
            patch = np.expand_dims(patch, 1) # Add channel dimension
            if i == 0:
                metadata = tif.imagej_metadata
                patch_size = ast.literal_eval(metadata['patch_size'])
                patch_stride = ast.literal_eval(metadata['patch_stride'])
                patches_structure = ast.literal_eval(metadata['patches_structure'])
                orig_img_shape = ast.literal_eval(metadata['original_img_shape'])
                print(metadata)
        imgs.append(patch)
    img_shape = np.array(patches_structure) * np.array(patch_size)
    n_patches = len(imgs)
    imgs = np.array(imgs)
    imgs = imgs.reshape(patches_structure + patch_size)
    img = np.squeeze(depatchify_merge(imgs, img_shape, patch_stride))

    print(f'\n{img.shape}')
    print('\n')


    # Depatchify binary
    binary_files = sorted([f for f in binary_path_sample.iterdir() if f.is_file() and f.suffix == '.tif' and not f.stem.startswith('.')])
    binary = []
    for i, f in enumerate(tqdm(binary_files, desc='Depatchifying binary', colour='blue', leave=True)):
        with tifffile.TiffFile(f) as tif:
            patch = tif.asarray().astype(np.float32)
            patch = np.expand_dims(patch, 1)
            if i == 0:
                metadata = tif.imagej_metadata
                patch_size = ast.literal_eval(metadata['patch_size'])
                patch_stride = ast.literal_eval(metadata['patch_stride'])
                patches_structure = ast.literal_eval(metadata['patches_structure'])
                orig_img_shape = ast.literal_eval(metadata['original_img_shape'])
                print(metadata)
        binary.append(patch)
    img_shape = np.array(patches_structure) * np.array(patch_size)
    binary = np.array(binary)
    binary = binary.reshape(patches_structure + patch_size)
    binary = np.squeeze(depatchify_merge(binary, img_shape, patch_stride))
    binary = (binary > 0.5).astype(np.uint8)
    if do_fill_holes:
        print('Filling binary holes and borders')
        binary = fill_binary_holes_border(binary)
    print(f'\n{binary.shape}')
    print('\n')

    # Depatchify predictions
    preds_files = sorted([f for f in preds_path_sample.iterdir() if f.is_file() and f.suffix == '.tif' and not f.stem.startswith('.')])
    preds = []
    for i, f in enumerate(tqdm(preds_files, desc='Depatchifying predictions', colour='blue', leave=True)):
        with tifffile.TiffFile(f) as tif:
            patch = tif.asarray().astype(np.float32)
            patch = np.expand_dims(patch, 1)
            # if i == 0:
            #     metadata = tif.imagej_metadata
            #     patch_size = ast.literal_eval(metadata['patch_size'])
            #     patch_stride = ast.literal_eval(metadata['patch_stride'])
            #     patches_structure = ast.literal_eval(metadata['patches_structure'])
            #     orig_img_shape = ast.literal_eval(metadata['original_img_shape'])
            #     print(metadata)
        preds.append(patch)
    img_shape = np.array(patches_structure) * np.array(patch_size)
    preds = np.array(preds)
    preds = preds.reshape(patches_structure + patch_size)
    pred = np.squeeze(depatchify_merge(preds, img_shape, patch_stride))

    print(f'\n{pred.shape}')
    print('\n')

    # Depatchify points
    coord_files = sorted([f for f in Path(coord_path_sample).iterdir() if f.is_file() and f.suffix == '.csv' and not f.stem.startswith('.')])
    coords = np.array([])
    for i, f in enumerate(tqdm(coord_files, desc='Depatchifying points', colour='blue', leave=True)):
        coord = pd.read_csv(f, header=0, index_col=0).values
        if len(patches_structure) == 3:
            patches_structure.insert(1, 1)
        if len(patch_stride) == 3:
            patch_stride.insert(1, 1)
        indices = np.unravel_index(i, patches_structure)
        coord[:,0] += indices[0]*patch_stride[0]
        coord[:,1] += indices[2]*patch_stride[2]
        coord[:,2] += indices[3]*patch_stride[3]
        coords = np.concatenate((coords, coord), axis=0) if len(coords) > 0 else coord

    coords = merge_close_points(coords, voxelsize=config['raw']['voxelsize'])
    if do_mask_points:
        coords = mask_points(coords, binary)
    print(f'\n{coords.shape}')
    print('\n')

    # Save
    print('Saving depatchified images and points')
    filename = output_path_sample.joinpath(input_path_sample.stem + '_img.tif')
    tifffile.imwrite(filename, img.astype(np.float32), compression='zlib')

    filename = output_path_sample.joinpath(input_path_sample.stem + '_binary.tif')
    tifffile.imwrite(filename, binary.astype(np.uint8), compression='zlib')

    filename = output_path_sample.joinpath(input_path_sample.stem + '_pred.tif')
    tifffile.imwrite(filename, pred.astype(np.float32), compression='zlib')

    filename = output_path_sample.joinpath(input_path_sample.stem + '_points.csv')
    points = pd.DataFrame(coords)
    if len(points.columns) == 3: # do not put header in empty predictions
        points.columns = ['axis-0', 'axis-1', 'axis-2']
    points.to_csv(filename, index=True, header=True)