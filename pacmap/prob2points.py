from argparse import ArgumentParser
from datetime import datetime
from tqdm import tqdm
from pathlib import Path
import tifffile
import numpy as np
import pandas as pd
from skimage.morphology import ball
from skimage.transform import resize

from .peaks import peak_local_max
from .utils import merge_close_points


def create_footprint(se_size, voxelsize):
    se = ball(se_size)
    se_size_vox = (se_size / np.array(voxelsize)).astype(int)
    se_size_vox = se_size_vox + (se_size_vox % 2 == 0) # Make sure that the size is odd
    footprint_vox = resize(se.astype(float), se_size_vox, order=1) > 0.5
    return footprint_vox


def get_points(probs, min_distance, threshold_abs, exclude_border=False, mask=None, p_norm=2, intensity_as_spacing=False, top_down=True, voxelsize=None):
    if voxelsize is None:
        footprint=min_distance
    else:
        footprint = create_footprint(min_distance, voxelsize)

    points = peak_local_max(
        probs,
        min_distance = min_distance,
        threshold_abs = threshold_abs,
        footprint = footprint,
        exclude_border = exclude_border,
        labels = mask,
        p_norm=p_norm,
        intensity_as_spacing=intensity_as_spacing,
        top_down=top_down,
        voxelsize=voxelsize,
        ).astype(int)
    return points


def main(**kwargs):
    input_path_probs = Path(kwargs['input_path_probs'])
    input_path_masks = Path(kwargs['input_path_masks'])
    batches_probs = sorted([batch for batch in input_path_probs.iterdir() if batch.is_dir()])
    if input_path_masks is not None:
        use_masks = True
        batches_masks = sorted([batch for batch in input_path_masks.iterdir() if batch.is_dir()])
        batches = [batch.stem for batch in batches_probs if batch.stem in [b.stem for b in batches_masks]]
        if not batches:
            raise ValueError('No corresponding batches found in probabilities and mask paths')
    else:
        use_masks = False
        batches = [batch.stem for batch in batches_probs]
        if not batches:
            raise ValueError('No batches found in probabilities path')
    
    output_path = Path(kwargs['output_path'])
    output_path.mkdir(parents=True, exist_ok=True)
    
    for batch in tqdm(batches, colour='green'):
        files_prob = sorted([f for f in input_path_probs.joinpath(batch).iterdir() if f.is_file() and f.suffix == '.tif'])
        if use_masks:
            files_binary = sorted([f for f in input_path_masks.joinpath(batch).iterdir() if f.is_file() and f.suffix == '.tif'])
            files = [f.name for f in files_prob if f.stem in [img.stem for img in files_binary]]
            if not files:
                raise ValueError('No corresponding filenames for grayscale and binary patches')
        else:
            files = [f.name for f in files_prob]

        for file in tqdm(files, colour='yellow'):
            filename_probs = input_path_probs.joinpath(batch, file)
            filename_masks = input_path_masks.joinpath(batch, file)
            probs = tifffile.imread(filename_probs)
            with tifffile.TiffFile(filename_probs) as tif:
                if tif.imagej_metadata:
                    metadata = tif.imagej_metadata
                else:
                    metadata = {}
                metadata['axes'] = 'ZYX'

            if use_masks:
                mask = tifffile.imread(filename_masks).astype(bool)
                probs = np.where(mask, probs, 0.0)
            
            # Get centroids
            points = get_points(
                probs, 
                min_distance=kwargs['min_distance'],
                threshold_abs=kwargs['threshold_abs'],
                exclude_border=kwargs['exclude_border'], 
                mask=mask if use_masks else None,
                p_norm=2,
                intensity_as_spacing=kwargs['intensity_as_spacing'],
                top_down=True if not kwargs['intensity_as_spacing'] or kwargs['top_down'] else False,
                voxelsize=kwargs['voxelsize']
            )

            if kwargs['merge_close_points']:
                points = merge_close_points(points, kwargs['voxelsize'], kwargs['min_distance'])

            # Save points
            filename = Path(file).stem
            if kwargs['save_points_npy']:
                filepath = output_path.joinpath('points', batch, filename + '.npy')
                filepath.parent.mkdir(parents=True, exist_ok=True)
                np.save(filepath, points)
            if kwargs['save_points_csv']:
                filepath = output_path.joinpath('points_csv', batch, filename + '.csv')
                filepath.parent.mkdir(parents=True, exist_ok=True)
                if len(points) == 0:
                    points = list() # to avoid error in pandas when having an empty array
                df = pd.DataFrame(points, columns=['axis-0', 'axis-1', 'axis-2'])
                df.to_csv(filepath, index=True, index_label='index')
            if kwargs['save_points_imgs']:
                filepath = output_path.joinpath('points_imgs', batch, filename + '.tif')
                filepath.parent.mkdir(parents=True, exist_ok=True)
                points_img = np.zeros_like(probs, dtype=bool)
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
    parser.add_argument('--input_path_probs', type=str, default='../data/02a_patches',
        help='Define input path with centroid probability image patches')
    parser.add_argument('--input_path_masks', type=str, default='../data/02b_patches_binarized',
        help='Define input path with foreground mask image patches')
    parser.add_argument('-o', '--output_path', type=str, default='../data/03_weak_targets',
        help='Define output path where the patches with weak targets are stored')
    parser.add_argument('-r', '--radi_um', nargs=3, type=float, default=[7.188, 7.188, 7.188],
        help='Define the radi in um for z, y, x for the gaussian kernel to place at each weak centroid in the targets')
    parser.add_argument('-v', '--voxelsize', nargs=3, type=float, default=[0.7188, 0.7188, 0.7188],
        help='Define the voxelsize in um for z, y, x')
    parser.add_argument('-m', '--min_distance', type=int, default=5,
        help='Define the minimum distance in pixels between weak centroids (default: 5). ' \
            'If voxelsize is defined, the distance is calculated in physical units. ' \
            'Otherwise, the distance is interpreted in voxels.')
    parser.add_argument('-t', '--threshold_abs', type=float, default=None,
        help='Define the absolute threshold for the centroid probabilities (default: None)')
    parser.add_argument('-b', '--exclude_border', nargs='*', default=True,
        help='Exclude local maxima within min_distance from the the border of the image (default: True).' \
            'If voxelsize is defined, the border is calculated in physical units. ' \
            'Otherwise, the border is interpreted in voxels.')
    parser.add_argument('--intensity_as_spacing', action='store_true',
        help='Use intensity as spacing')
    parser.add_argument('--top_down', action='store_true',
        help='Use top down approach for peak detection')
    parser.add_argument('--merge_close_points', type=bool, default=True,
        help='Merge points that are too close')
    parser.add_argument('--save_points_npy', action='store_true',
        help= 'Store weak centroids as z, y, x coords in .npy files')
    parser.add_argument('--save_points_csv', action='store_true',
        help= 'Store weak centroids as z, y, x coords in .csv files')
    parser.add_argument('--save_points_imgs', action='store_true',
        help= 'Store weak centroids as boolean array in .tif files')
    args, _ = parser.parse_known_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    main(**vars(args))