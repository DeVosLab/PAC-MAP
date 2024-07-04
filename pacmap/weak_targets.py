from argparse import ArgumentParser
from datetime import datetime
from tqdm import tqdm
from pathlib import Path
import tifffile
import numpy as np
import pandas as pd
from skimage.filters import threshold_niblack as threshold
from skimage.morphology import binary_opening
from skimage.segmentation import watershed
from skimage.measure import regionprops_table
from scipy.ndimage import distance_transform_edt, gaussian_filter, label, median_filter
from scipy.signal import gaussian

from .peaks import peak_local_max


def __get_gaussian_kernel(r_z, r_y, r_x):
    '''Create a 3D Gaussian kernel with the given radii and standard deviation
    
    Parameters
    ----------
    r_z : int
        Radius of the kernel in the z-axis
    r_y : int
        Radius of the kernel in the y-axis
    r_x : int
        Radius of the kernel in the x-axis

    Returns
    -------
    gauss : np.array
        3D Gaussian kernel
    '''
    gkern1d_z = gaussian(r_z*2 + 1, std=r_z/4) # +1 to have an odd number of elements
    gkern1d_y = gaussian(r_y*2 + 1, std=r_y/4)
    gkern1d_x = gaussian(r_x*2 + 1, std=r_x/4)
    gauss = np.einsum('i,j,k',gkern1d_z,gkern1d_y,gkern1d_x)
    gauss = (gauss - gauss.min())/(gauss.max() - gauss.min()) 
    return gauss


def create_target_img(points, img_shape, radi, values=1):
    '''Create a 3D image with Gaussian kernels placed at the given points

    Parameters
    ----------
    points : np.array
        Array with the coordinates of the points to place the kernels
    img_shape : tuple
        Shape of the image in which the kernels will be placed
    radi : list
        List with the radii of the kernel in the z, y, x axes
    values : int, float or np.array
        Amplitude of the Gaussian kernel at each point. If int or float, the same value is used for all points. Default is 1.

    Returns
    -------
    temp_img : np.array
        3D image with the Gaussian kernels placed at the given points
    '''

    if len(points) == 0:
        return np.zeros(img_shape)
    if values is None:
        values = np.ones(len(points))
    if isinstance(values, (int, float)):
        values = np.ones(len(points)) * values
    
    assert len(points) == len(values), 'The number of points and values must be the same'

    D, H, W = img_shape
    r_z, r_y, r_x = radi
    kernel = __get_gaussian_kernel(r_z, r_y, r_x)

    # Create a temporary image to place the kernels 
    temp_img = np.zeros((D+2*r_z,H+2*r_y, W+2*r_x)) # Add padding to allow kernels to be placed at the border

    # Loop over points and place kernels in the image
    for point, value in zip(points, values):
        # If value is inf, continue
        if np.isinf(value):
            continue

        z = int(point[0])
        y = int(point[1])
        x = int(point[2])
        
        # Get kernel values at the current point
        current_patch = temp_img[
            r_z+z-r_z:r_z+z+r_z,  # Add r_z/y/x to get the correct position in the padded image
            r_y+y-r_y:r_y+y+r_y,
            r_x+x-r_x:r_x+x+r_x
        ]

        current_patch_shape = current_patch.shape
        current_kernel = value * kernel[
            :current_patch_shape[0],
            :current_patch_shape[1],
            :current_patch_shape[2]
        ]

        # Place the kernel in the image, keeping the maximum value
        temp_img[
            r_z+z-r_z:r_z+z+r_z,
            r_y+y-r_y:r_y+y+r_y,
            r_x+x-r_x:r_x+x+r_x
        ] = np.maximum(current_patch, current_kernel)

    # Remove padding
    temp_img = temp_img[
        r_z:-r_z,
        r_y:-r_y,
        r_x:-r_x
    ]
    return temp_img


def main(**kwargs):
    '''Create weak targets using the distance transform and watershed algorithm'''

    # Define input paths
    input_path_grayscale = Path(kwargs['input_path_grayscale'])
    input_path_binary = Path(kwargs['input_path_binary'])

    # Get batches (subfolders) that are in both grayscale and binary paths
    batches_grayscale = sorted([batch for batch in input_path_grayscale.iterdir() if \
                                batch.is_dir() and not batch.stem.startswith('.')])
    batches_binary = sorted([batch for batch in input_path_binary.iterdir() if \
                             batch.is_dir() and not batch.stem.startswith('.')])

    batches = [batch.stem for batch in batches_grayscale if batch.stem in [b.stem for b in batches_binary]]
    if not batches:
        raise ValueError('No corresponding batches in grayscale and binary paths')
    
    # Define output path
    output_path = Path(kwargs['output_path'])
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Loop over batches
    for batch in tqdm(batches, colour='green'):
        # Get filenames in batch
        files_grayscale = sorted([f for f in input_path_grayscale.joinpath(batch).iterdir() if \
                                  f.is_file() and f.suffix == '.tif' and not f.stem.startswith('.')])
        files_binary = sorted([f for f in input_path_binary.joinpath(batch).iterdir() if \
                               f.is_file() and f.suffix == '.tif' and not f.stem.startswith('.')])
        files = [f.name for f in files_grayscale if f.stem in [img.stem for img in files_binary]]
        if not files:
            raise ValueError('No corresponding filenames for grayscale and binary patches')

        # Loop over files
        for file in tqdm(files, colour='yellow'):
            # Load images
            filename_grayscale = input_path_grayscale.joinpath(batch, file)
            filename_binary = input_path_binary.joinpath(batch, file)
            patch = tifffile.imread(filename_grayscale)
            foreground = tifffile.imread(filename_binary).astype(bool)

            # Get metadata from grayscale image
            with tifffile.TiffFile(filename_grayscale) as tif:
                if tif.imagej_metadata:
                    metadata = tif.imagej_metadata
                else:
                    metadata = {}
                metadata['axes'] = 'ZYX'

            # Threshold the grayscale image and mask using the foreground
            patch_bw = patch > threshold(patch)
            patch_bw = np.where(foreground, patch_bw, False)

            # Apply morphological operations to the binary image
            se_size = 1 # in physical units (um)
            se_size_vox = np.ceil(se_size / np.array(kwargs['voxelsize'])).astype(int)
            patch_bw = binary_opening(
                patch_bw,
                footprint=np.ones(se_size_vox)
                )
            patch_bw = median_filter(patch_bw, size=se_size_vox)

            # Calculate distance transform
            distance = gaussian_filter(distance_transform_edt(patch_bw), se_size_vox)
            
            # Find local maxima
            coords = peak_local_max(
                distance,
                min_distance = kwargs['min_distance'],
                footprint = kwargs['min_distance'],
                exclude_border = kwargs['exclude_border'],
                labels = patch_bw,
                voxelsize=kwargs['voxelsize'],
                ).astype(int)

            # Apply watershed algorithm
            mask = np.zeros(distance.shape, dtype=bool)
            mask[tuple(coords.T)] = True
            markers, _ = label(mask)
            labels = watershed(-distance, markers, mask=patch_bw)
            labels = np.where(foreground, labels, 0)
            
            # Get centroids of the weak targets
            props = regionprops_table(labels, properties=('label', 'centroid'))
            points = np.round(np.array([[z, y, x] for z, y, x in zip(
            props['centroid-0'],
            props['centroid-1'],
            props['centroid-2']
            )])).astype(int)

            # Save images and points
            filename = Path(file).stem
            if kwargs['save_binary']:
                filepath = output_path.joinpath('binary', batch, filename + '.tif')
                filepath.parent.mkdir(parents=True, exist_ok=True)
                tifffile.imwrite(
                    filepath,
                    patch_bw.astype(np.float32),
                    imagej=True,
                    metadata=metadata,
                    compression='zlib'
                )
            if kwargs['save_labels']:
                filepath = output_path.joinpath('labels', batch, filename + '.tif')
                filepath.parent.mkdir(parents=True, exist_ok=True)
                tifffile.imwrite(
                    filepath,
                    labels.astype(np.uint16),
                    imagej=True,
                    metadata=metadata,
                    compression='zlib'
                )
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
                points_img = np.zeros(patch.shape, dtype=bool)
                points_img[tuple(points.T)] = True
                tifffile.imwrite(
                    filepath,
                    points_img.astype(np.float32),
                    imagej=True,
                    metadata=metadata,
                    compression='zlib'
                )
            if kwargs['save_targets']:
                filepath = output_path.joinpath('targets', batch, filename + '.tif')
                filepath.parent.mkdir(parents=True, exist_ok=True)
                radi_um = kwargs['radi_um']
                voxelsize = kwargs['voxelsize']
                radi_vox = [int(r/v) for r,v in zip(radi_um, voxelsize)]
                target = create_target_img(
                    points,
                    patch.shape,
                    radi_vox
                    ).astype(float)
                tifffile.imwrite(
                    filepath,
                    target.astype(np.float32),
                    imagej=True,
                    metadata=metadata,
                    compression='zlib'
                )
        
        
def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--input_path_grayscale', type=str, default='../data/02a_patches',
        help='Define input path with grayscale image patches')
    parser.add_argument('--input_path_binary', type=str, default='../data/02b_patches_binarized',
        help='Define input path with binary foreground image patches')
    parser.add_argument('-o', '--output_path', type=str, default='../data/03_weak_targets',
        help='Define output path where the patches with weak targets are stored')
    parser.add_argument('-r', '--radi_um', nargs=3, type=float, default=[7.188, 7.188, 7.188],
        help='Define the radi in um for z, y, x for the gaussian kernel to place at each weak centroid in the targets')
    parser.add_argument('-v', '--voxelsize', nargs=3, type=float, default=[0.7188, 0.7188, 0.7188],
        help='Define the voxelsize in um for z, y, x')
    parser.add_argument('-m', '--min_distance', type=int, default=5,
        help='Define the minimum distance in pixels between weak centroids (default: 10). ' \
            'If voxelsize is defined, the distance is calculated in physical units. ' \
            'Otherwise, the distance is interpreted in voxels.')
    parser.add_argument('-b', '--exclude_border', nargs='*', default=True,
        help='Exclude local maxima in the border of the image (default: True).' \
            'If voxelsize is defined, the border is calculated in physical units. ' \
            'Otherwise, the border is interpreted in voxels.')
    parser.add_argument('--save_binary', action='store_true',
        help='Store binary images of foreground in .tif files')
    parser.add_argument('--save_labels', action='store_true',
        help='Store weak labels images of detected nuclei in .tif files')
    parser.add_argument('--save_points_npy', action='store_true',
        help= 'Store weak centroids as z, y, x coords in .npy files')
    parser.add_argument('--save_points_csv', action='store_true',
        help= 'Store weak centroids as z, y, x coords in .csv files')
    parser.add_argument('--save_points_imgs', action='store_true',
        help= 'Store weak centroids as boolean array in .tif files')
    parser.add_argument('--save_targets', action='store_true',
        help= 'Store weak centroid targets as float array in .tif files')
    args, _ = parser.parse_known_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    main(**vars(args))