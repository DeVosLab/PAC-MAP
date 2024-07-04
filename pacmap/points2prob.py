from argparse import ArgumentParser
from dotenv import load_dotenv
import os
import numpy as np
import pandas as pd
from pathlib import Path
import tifffile
from scipy.spatial import KDTree

from .weak_targets import create_target_img
from .points2df import get_points_by_method


def main(**kwargs):
    # Load environment variables
    load_dotenv()
    data_path = Path(os.getenv('DATA_PATH'))

    #Set settings
    input_path = data_path.joinpath(kwargs['input_path'])
    output_path = data_path.joinpath(kwargs['output_path'])
    img_path = data_path.joinpath(kwargs['img_path'])
    radi_um = kwargs['radi_um']
    voxelsize = kwargs['voxelsize']
    img_shape = kwargs['img_shape']

    method2suffix = {
        'masks': '.tif',
        'npy': '.npy',
        'csv': '.csv'
    }
    suffix = method2suffix[kwargs['method']]

    #Iterate over samples and get patch annotation files
    if kwargs['check_batches']:
        samples = sorted([f for f in input_path.iterdir() if f.is_dir()])
        img_paths = [img_path.joinpath(s.stem) for s in samples] if \
                     img_path is not None else None
    else:
        samples = [input_path]
        img_paths = [img_path] if img_path is not None else None

    for i, sample in enumerate(samples):
        if img_paths is not None:
            sample_img_path = img_paths[i]

        files = sorted([f for f in sample.iterdir() if \
                        f.is_file() and f.suffix == suffix and not f.stem.startswith('.')])
        for _, file in enumerate(files):
            print(file)

            #Get coordinates from annotation file
            points, shape = get_points_by_method(file, kwargs['method'])
            points = points.astype(int)

            if shape is not None:
                img_shape = shape
            elif img_paths is not None and sample_img_path.joinpath(file.stem + '.tif').is_file():
                img_file = sample_img_path.joinpath(file.stem + '.tif')
                img = tifffile.imread(img_file)
                img_shape = img.shape

            if kwargs['save_points_csv']:
                filename = output_path.joinpath(
                    'points_csv',
                    sample.stem if kwargs['check_batches'] else '',
                    file.stem + '.csv'
                    )
                filename.parent.mkdir(parents=True, exist_ok=True)
                if len(points) == 0:
                    points = list() # to avoid error in pandas when having an empty array
                df = pd.DataFrame(points, columns=['axis-0', 'axis-1', 'axis-2'])
                df.to_csv(filename, index=True, index_label='index')
            
            if kwargs['save_targets']:

                if kwargs['intensity_as_spacing']:
                    kdtree_pos = KDTree(points * voxelsize)
                    dist, _ = kdtree_pos.query(
                        points * voxelsize,
                        k=2
                        )
                    dist = dist[:,1]
                    # If dist is inf, continue
                    if np.any(np.isinf(dist)):
                        continue

                # Add probability kernels at points
                radi_vox = [int(r/v) for r,v in zip(radi_um, voxelsize)]
                targets = create_target_img(
                    points,
                    img_shape,
                    radi_vox,
                    values=dist if kwargs['intensity_as_spacing'] else None,
                    ).astype(float)
                
                # Write to tiffile
                filename = output_path.joinpath(
                    'targets',
                    sample.stem if kwargs['check_batches'] else '',
                    file.stem + '.tif')
                filename.parent.mkdir(parents=True, exist_ok=True)
                tifffile.imwrite(
                    filename,
                    targets.astype(np.float32),
                    imagej=True,
                    compression='zlib'
                )


def parse_args():
    parser = ArgumentParser(description='Convert points to targets.')
    parser.add_argument('-i', '--input_path', type=str, default='manual_annotation/03a_manual_annotations/all/points_csv',
                        help='Input path to csv files with point coordinates.')
    parser.add_argument('-o', '--output_path', type=str, default='manual_annotation/03a_manual_annotations/all/targets',
                        help='Output path to where targets tiff files will be written.')
    parser.add_argument('-s', '--img_shape', type=int, nargs=3, default=None,
                        help='Define the size of the image volume in pixels for Z, Y, X.')
    parser.add_argument('--img_path', type=str, default=None,
                        help='Define the path to the image volumes from which img shapes are infered')
    parser.add_argument('--intensity_as_spacing', action='store_true',
                        help='Use intensity of probability kernel as measure for spacing between closest points.')
    parser.add_argument('-v', '--voxelsize', type=float, nargs=3, default=[1.9999, 0.3594, 0.3594],
                        help='Define the voxelsize of the image volume in Z, Y, X.')
    parser.add_argument('-r', '--radi_um', type=float, nargs=3, default=[10, 10, 10],
                        help='Define the size of the probability kernel in physical units')
    parser.add_argument('--method', type=str, choices=('masks', 'npy', 'csv'), default='csv',
                        help='Define the method used to store the points')
    parser.add_argument('--check_batches', action='store_true',
                        help='Check if input_path contains multiple batches')
    parser.add_argument('--save_points_csv', action='store_true',
                        help= 'Store weak centroids as z, y, x coords in .csv files')
    parser.add_argument('--save_targets', action='store_true',
                        help= 'Store weak centroid targets as float array in .tif files')
    args, _ = parser.parse_known_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    main(**vars(args))