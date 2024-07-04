from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm
import numpy as np
import tifffile
from skimage.measure import regionprops_table
import pandas as pd

from .utils import merge_close_points


def get_points_by_method(file, method):
    if method == 'npy':
        points = np.load(file)
        return points, None
    elif method == 'csv':
        csv = pd.read_csv(file, header=0, index_col=0)
        if not csv.empty:
            points = np.array(csv.values)
        elif csv.empty:
            points = np.zeros(csv.shape)
        return points, None
    elif method == 'masks':
        masks = tifffile.imread(file)
        props = regionprops_table(masks, properties=('centroid',))
        points = np.array([[z,y,x] for (z,y,x) in zip(props['centroid-0'],props['centroid-1'],props['centroid-2'])])
        return points, masks.shape


def remove_border_points(points, voxelsize, border_size_um, volumesize_um):
    dim_min = border_size_um
    dim_max = volumesize_um - border_size_um
    points = points[
        ((points * voxelsize > dim_min) & (points * voxelsize < dim_max)).all(axis=1)
        ]
    return points


def main(**kwargs):

    assert kwargs['input_paths'] is not None, \
        'input_paths must be defined'
    assert kwargs['methods'] is not None, \
        'methods must be defined'
    assert kwargs['output_path'] is not None, \
        'output_path must be defined'
    
    if isinstance(kwargs['input_paths'], str):
        kwargs['input_paths'] = [kwargs['input_paths']]
    if isinstance(kwargs['methods'], str):
        kwargs['methods'] = [kwargs['methods']]
    if isinstance(kwargs['use_mask'], bool):
        kwargs['use_mask'] = [kwargs['use_mask']]

    assert len(kwargs['input_paths']) == len(kwargs['methods']), \
        'Number of input_paths and methods should be the same'

    if len(kwargs['input_paths']) > len(kwargs['use_mask']):
        assert len(kwargs['use_mask']) == 1, \
            'If use_mask is not defined for all input_paths, only one value should be defined'
        kwargs['use_mask'] = kwargs['use_mask'] * len(kwargs['input_paths'])

    assert len(kwargs['input_paths']) == len(kwargs['use_mask']), \
        'Number of input_paths and use_mask should be the same'
    
    if any(kwargs['use_mask']):
        assert kwargs['mask_path'] is not None, \
            'If use_mask is True, mask_path should be defined'

    input_paths = [Path(i) for i in kwargs['input_paths']]
    methods = kwargs['methods']
    
    method2suffix = {
        'masks': '.tif',
        'npy': '.npy',
        'csv': '.csv'
    }

    output_path = Path(kwargs['output_path'])
    output_path.mkdir(parents=True, exist_ok=True)

    # Get the samples that are present in all the folders
    df_samples = pd.DataFrame(columns=['input_path', 'sample'])
    for input_path, method in zip(input_paths, methods):
        if kwargs['check_batches']:
            df_samples = pd.concat([df_samples, pd.DataFrame(
                [[input_path, sample.stem] for sample in input_path.iterdir() if sample.is_dir()],
                columns=['input_path', 'sample']
                )], ignore_index=True)
        else:
            df_samples = pd.concat([df_samples, pd.DataFrame(
                {'input_path':input_path, 'sample': input_path.stem}, index=[0]
                )], ignore_index=True)

    df_samples = df_samples.groupby('sample').filter(lambda x: len(x) == len(input_paths))
    df_samples = df_samples.reset_index(drop=True)

    if kwargs['mask_path'] is not None:
        mask_path = Path(kwargs['mask_path'])
        if kwargs['check_batches']:
            samples_mask = sorted([sample.stem for sample in mask_path.iterdir() if sample.is_dir()])
        else:
            samples_mask = [mask_path.stem]
        df_samples = df_samples[df_samples['sample'].isin(samples_mask)]
    
    # Get the files for each sample that are present in all the folders
    df_files = pd.DataFrame(columns=['input_path', 'sample', 'file'])
    for input_path, sample in zip(df_samples['input_path'], df_samples['sample']):
        # Get the method based on the input path
        method = methods[input_paths.index(input_path)]

        # Add all files of each sample
        if kwargs['check_batches']:
            df_files = pd.concat([df_files, pd.DataFrame(
                [[input_path, sample, file.stem] for \
                file in input_path.joinpath(sample).iterdir() if \
                file.is_file() and file.suffix == method2suffix[method] and \
                not file.stem.startswith('.')],
                columns=['input_path', 'sample', 'file']
                )], ignore_index=True)
        else:
            df_files = pd.concat([df_files, pd.DataFrame(
                [[input_path, sample, file.stem] for \
                file in input_path.iterdir() if \
                file.is_file() and file.suffix == method2suffix[method] and \
                not file.stem.startswith('.')],
                columns=['input_path', 'sample', 'file']
                )], ignore_index=True)
    df_files = df_files.groupby(['sample', 'file']).filter(lambda x: len(x) == len(input_paths))
    df_files = df_files.reset_index(drop=True)
    
    if kwargs['mask_path'] is not None:
        if kwargs['check_batches']:
            mask_files = sorted([f.stem for f in mask_path.joinpath(sample).iterdir() if \
                             f.is_file() and f.suffix == method2suffix['masks'] and \
                                not f.stem.startswith('.')])
        else:
            mask_files = sorted([f.stem for f in mask_path.iterdir() if \
                                f.is_file() and f.suffix == method2suffix['masks'] and \
                                    not f.stem.startswith('.')])
        df_files = df_files[df_files['file'].isin(mask_files)]
    
    voxelsize = np.array(kwargs['voxelsize'])
    volumesize_vox = kwargs['volumesize_vox']
    bordersize_um = kwargs['bordersize_um']
    volumesize_um = voxelsize * np.array(volumesize_vox) if volumesize_vox is not None else None

    columns = ['dataset', 'input_path', 'sample', 'file', 'axis-0', 'axis-1', 'axis-2']
    df = pd.DataFrame(columns=columns)
    dataset_names = kwargs['dataset_names']
    use_mask = kwargs['use_mask']
    merge_distance_um = kwargs['merge_distance_um']
    if isinstance(merge_distance_um, (int, float)):
        merge_distance_um = [merge_distance_um] * len(dataset_names)

    for dataset_name, input_path, method, do_mask, merge_dist in zip(dataset_names, input_paths, methods, use_mask, merge_distance_um):
        df_files_input_path = df_files[df_files['input_path'] == input_path]
        for sample, files in df_files_input_path.groupby('sample')['file']:
            for file in files:
                # Load points by method
                suffix = method2suffix[method]
                if kwargs['check_batches']:
                    filename = input_path.joinpath(sample, file + suffix)
                else:
                    filename = input_path.joinpath(file + suffix)
                points, _ = get_points_by_method(
                    filename, 
                    method
                    )
                if merge_dist > 0:
                    points = merge_close_points(points, voxelsize, merge_dist)
                points = points.astype(int)
                
                # Remove border points
                if bordersize_um is not None and volumesize_um is not None:
                    points = remove_border_points(points, voxelsize, bordersize_um, volumesize_um)
                
                # Remove points outside of mask
                if do_mask:
                    mask = tifffile.imread(mask_path.joinpath(sample, file + '.tif')).astype(bool)
                    points_img = np.zeros(mask.shape, dtype=bool)
                    points_img[tuple(points.T)] = True
                    points_img = np.where(mask, points_img, False)
                    points = np.stack((np.where(points_img)), axis=1)

                # Add points to df as separate rows
                n_points = points.shape[0]
                if n_points == 0:
                    continue

                df = pd.concat([df, pd.DataFrame(
                    np.concatenate((
                        np.repeat(dataset_name, n_points)[:,None],
                        np.repeat(str(input_path), n_points)[:,None],
                        np.repeat(sample, n_points)[:,None],
                        np.repeat(file, n_points)[:,None],
                        points
                        ), axis=1),
                    columns=columns
                    )], ignore_index=True)
    
    # Save df
    df['dataset'] = df['dataset'].astype('category')
    df['input_path'] = df['input_path'].astype('category')
    df['sample'] = df['sample'].astype('category')
    df['file'] = df['file'].astype('category')
    df['axis-0'] = df['axis-0'].astype(int)
    df['axis-1'] = df['axis-1'].astype(int)
    df['axis-2'] = df['axis-2'].astype(int)
    df.to_csv(output_path.joinpath('points.csv'), index=False)

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--dataset_names', type=str, nargs='+', default=None,
        help='Provide names of the datasets, i.e., one for each input path')
    parser.add_argument('--input_paths', type=str, nargs='+', default=None,
        help='Define path(s) where the points data is located')
    parser.add_argument('--methods', type=str, nargs='+', choices=['masks', 'npy', 'csv'], default=None,
        help='Define method(s) by which the points are stored')
    parser.add_argument('--mask_path', type=str, default=None,
        help='Define path where binary mask patches are located to mask points.')
    parser.add_argument('--use_mask', type=bool, default=False,
        help='Define whether to use masks to mask predictions. Default: False.'
             'If True, mask_path must be defined.')
    parser.add_argument('--output_path', type=str, default=None,
        help='Define the output path where results.csv file will be stored (optional)')
    parser.add_argument('--voxelsize', nargs=3, type=float, default=[0.7188, 0.7188, 0.7188],
        help='Define the voxel size (Z, Y, X) to use. The same voxel size is used for all datasets.')
    parser.add_argument('--volumesize_vox', nargs=3, type=int, default=None,
        help='Define the size of the volume (Z,Y,X) in voxels. The same volume size is used for all datasets.')
    parser.add_argument('--bordersize_um', type=float, default=None,
        help='Define the border size (in um) to be ignored in the evaluation')
    parser.add_argument('--check_batches', type=bool, default=False,
        help='Define whether to check batches in input_paths. Default: False.')
    parser.add_argument('--merge_distance_um', type=float, nargs='+', default=-1.0,
        help='Define the distance in um to merge close points. Default: -1.0 (no merging).')
    args, _ = parser.parse_known_args()  

    return args

if __name__ == '__main__':
    args = parse_args()
    main(**vars(args))