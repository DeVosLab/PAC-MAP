from argparse import ArgumentParser
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
import gc
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

def main(**kwargs):
    input_path = Path(kwargs['input_path'])
    binary_path = Path(kwargs['binary_path'])
    preds_path = Path(kwargs['preds_path'])
    coords_path = Path(kwargs['coords_path'])
    output_path = Path(kwargs['output_path'])
    output_path.mkdir(exist_ok=True, parents=True)

    samples = [f.name for f in input_path.iterdir() if f.is_dir()]
    for sample in tqdm(samples, desc='Depatchifying samples', colour='green', leave=True):
        input_path_sample = input_path.joinpath(sample)
        binary_path_sample = binary_path.joinpath(sample)
        preds_path_sample = preds_path.joinpath(sample)
        coords_path_sample = coords_path.joinpath(sample)
        output_path_sample = output_path.joinpath(sample)
        output_path_sample.mkdir(exist_ok=True, parents=True)

        # Images
        if kwargs['input_path'] is not None:
            ## Depatchify
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

            if kwargs['remove_zero_padding']:
                idxs = np.argwhere(img[:,0,:,:] > 0)
                z_min, y_min, x_min = idxs.min(axis=0)
                z_max, y_max, x_max = idxs.max(axis=0)
                img = img[z_min:z_max+1, :, y_min:y_max+1, x_min:x_max+1]

            print(f'\n Image shape: {img.shape}')
            print('\n')
            ## Save
            filename = output_path_sample.joinpath('input', input_path_sample.name).with_suffix('.tif')
            filename.parent.mkdir(exist_ok=True, parents=True)
            tifffile.imwrite(filename, img.astype(np.float32), imagej=True, metadata={'axes': 'ZCYX'}, compression='zlib')
            del img
            del imgs
            gc.collect()

        # Binary
        if kwargs['binary_path'] is not None:
            ## Depatchify
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
            if kwargs['do_fill_holes']:
                print('Filling binary holes and borders')
                binary = fill_binary_holes_border(binary)
            if kwargs['remove_zero_padding']:
                binary = binary[z_min:z_max+1, y_min:y_max+1, x_min:x_max+1]
            
            print(f'\n Binary shape: {binary.shape}')
            print('\n')
            ## Save
            filename = output_path_sample.joinpath('binary', binary_path_sample.name).with_suffix('.tif')
            filename.parent.mkdir(exist_ok=True, parents=True)
            tifffile.imwrite(filename, binary.astype(np.uint8), imagej=True, metadata={'axes': 'ZYX'}, compression='zlib')
            if not kwargs['do_mask_points']:
                del binary
                gc.collect()

        # Predictions
        if kwargs['preds_path'] is not None:
            ## Depatchify
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
            if kwargs['remove_zero_padding']:
                pred = pred[z_min:z_max+1, y_min:y_max+1, x_min:x_max+1]
            
            print(f'\n Preds shape: {pred.shape}')
            print('\n')
            ## Save
            filename = output_path_sample.joinpath('preds', preds_path_sample.name).with_suffix('.tif')
            filename.parent.mkdir(exist_ok=True, parents=True)
            tifffile.imwrite(filename, pred.astype(np.float32), imagej=True, metadata={'axes': 'ZYX'}, compression='zlib')
            del pred
            del preds
            gc.collect()

        # Points
        if kwargs['coords_path'] is not None:
            ## Depatchify
            coord_files = sorted([f for f in Path(coords_path_sample).iterdir() if f.is_file() and f.suffix == '.csv' and not f.stem.startswith('.')])
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
            coords = merge_close_points(coords, voxelsize=kwargs['voxelsize'])
            if kwargs['remove_zero_padding']:
                coords = coords[(coords[:,0] >= z_min) & (coords[:,0] <= z_max) & \
                                (coords[:,1] >= y_min) & (coords[:,1] <= y_max) & \
                                (coords[:,2] >= x_min) & (coords[:,2] <= x_max), :]
                coords = coords - np.array([z_min, y_min, x_min])
            
            if kwargs['do_mask_points']:
                coords = mask_points(coords, binary)
            print(f'\n Points shape: {coords.shape}')
            print('\n')
            ## Save
            filename = output_path_sample.joinpath('points', coords_path_sample.name).with_suffix('.csv')
            filename.parent.mkdir(exist_ok=True, parents=True)
            points = pd.DataFrame(coords)
            if len(points.columns) == 3: # do not put header in empty predictions
                points.columns = ['axis-0', 'axis-1', 'axis-2']
            points.to_csv(filename, index=True, header=True)

def parse_args():
    parser = ArgumentParser(description='Depatchify a dataset')
    parser.add_argument('-i', '--input_path', type=str, default=None, help='Path to the input dataset')
    parser.add_argument('-b', '--binary_path', type=str, default=None, help='Path to the binary dataset')
    parser.add_argument('-p', '--preds_path', type=str, default=None, help='Path to the predictions')
    parser.add_argument('-c', '--coords_path', type=str, default=None, help='Path to the coordinates')
    parser.add_argument('-o', '--output_path', type=str, default=None, help='Path to the output dataset')
    parser.add_argument('-f', '--do_fill_holes', type=bool, default=True, help='Fill holes in the binary dataset')
    parser.add_argument('-m', '--do_mask_points', type=bool, default=False, help='Mask points in the coordinates')
    parser.add_argument('-r', '--remove_zero_padding', type=bool, default=False, help='Remove zero padding from the dataset')
    parser.add_argument('-v', '--voxelsize', type=float, default=None, help='Voxel size')
    args, _ = parser.parse_known_args()
    return args

def check_args(args):
    if args.output_path is None:
        raise ValueError('output_path is required')
    if args.input_path is None and args.binary_path is None and args.preds_path is None and args.coords_path is None:
        raise ValueError('At least one of input_path, binary_path, preds_path or coords_path is required')
    if args.voxelsize is None and args.coords_path is not None:
        raise ValueError('voxelsize is required when coords_path is provided')

if __name__ == '__main__':
    args = parse_args()
    main(**vars(args))

