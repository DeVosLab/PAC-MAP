from argparse import ArgumentParser
import types
from pathlib import Path
from tqdm import tqdm
import tifffile
import numpy as np
import pandas as pd

from .utils import rescale_voxels, normalize_per_channel, make_dict_json_compatible


def main(**kwargs):
    input_path = Path(kwargs['input_path'])
    output_path = Path(kwargs['output_path'])
    if ~output_path.is_dir():
        output_path.mkdir(parents=True, exist_ok=True)

    # Insert element to voxelsizes for channel dim
    current_voxelsize = list(kwargs['current_voxelsize'])
    target_voxelsize = list(kwargs['target_voxelsize'])
    channel_dim = 1
    current_voxelsize.insert(channel_dim, 1)
    target_voxelsize.insert(channel_dim, 1)

    # Create channel to index mapping, since channels2store might not be in order or complete
    channel_mapping = {v: i for i, v in enumerate(kwargs['channels2store'])}

    # Define channels to normalize
    if kwargs['channels2normalize'] is not None:
        channels2normalize = [channel_mapping[c] for c in kwargs['channels2normalize']]
    else:
        channels2normalize = None

    # Get all filenames in input path
    filenames = sorted([f for f in input_path.iterdir() if \
        f.is_file() and f.suffix == kwargs['file_extension'] and \
        not f.stem.startswith('.')])

    # Loop over all files
    for filename in tqdm(filenames):
        
        # Load image
        if kwargs['verbose']:
            print(f'\nProcessing {filename}')
            print('Loading channels')
        img = tifffile.imread(filename).astype(np.float32)

        with tifffile.TiffFile(filename) as tif:
            if tif.imagej_metadata:
                metadata = tif.imagej_metadata
            else:
                metadata = {}

        # Select channels, add channel dim if necessary
        if kwargs['verbose']:
            print(f'Original image shape: {img.shape}')
        if img.ndim == 4:
            img = img[:,np.r_[kwargs['channels2store']],]
            crop_channel = channel_mapping[kwargs['crop_channel']]
        if img.ndim == 3:
            img = np.expand_dims(img, channel_dim)
            crop_channel = 0

        # Rescale voxels
        if kwargs['verbose']:
            print('Rescaling voxels')     
        img = rescale_voxels(img, current_voxelsize, target_voxelsize)
        if kwargs['verbose']:
            print(f'Rescaled image shape: {img.shape}')
        
        # Crop image
        if kwargs['verbose']:
            print('Custom crop function')
        if kwargs['crop_func'] is not None:
            try:
                img, bbox, foreground = kwargs['crop_func'](
                    img,
                    channel2use=crop_channel,
                    padding=kwargs["padding"],
                    min_size=kwargs["min_size"],
                    masked_patch=kwargs['masked_patch']
                    )
            except:
                print('Custom crop function failed. Probably no foreground found.')
                continue
        if kwargs['verbose']:
            print(f'Cropped image shape: {img.shape}')

        # Normalize image
        if kwargs['verbose']:
            print('Normalization')
        n_channels = len(kwargs['channels2store'])
        pmins, pmaxs = kwargs['pmins'], kwargs['pmaxs']
        pmins = n_channels*pmins if len(pmins) == 1 else pmins
        pmaxs = n_channels*pmaxs if len(pmaxs) == 1 else pmaxs
        img = normalize_per_channel(
            img,
            pmins=pmins,
            pmaxs=pmaxs,
            channels2normalize=channels2normalize
            )
        
        # Store preprocessed image
        if kwargs['verbose']:
            print('Storing output')
        metadata['spacing'] = target_voxelsize[0]
        metadata['axes'] = 'ZCYX'
        use_bigtiff = img.nbytes > 2**32
        tifffile.imwrite(
            output_path.joinpath(filename.name),
            img.astype(np.float32),
            imagej=not use_bigtiff,
            resolution=tuple(1/v for v in target_voxelsize[2:]),
            metadata=metadata if not use_bigtiff else make_dict_json_compatible(metadata),
            compression='zlib',
            bigtiff=use_bigtiff
            )
        
        # Store binary image of foreground
        if kwargs['output_path_binary'] is not None:
            output_path_binary = Path(kwargs['output_path_binary'])
            if ~output_path_binary.is_dir():
                output_path_binary.mkdir(parents=True, exist_ok=True)
            foreground = foreground.astype(np.float32)
            use_bigtiff = foreground.nbytes > 2**32
            tifffile.imwrite(
                output_path_binary.joinpath(filename.name),
                foreground.astype(np.float32),
                imagej=not use_bigtiff,
                resolution=tuple(1/v for v in target_voxelsize[2:]),
                metadata={'axes': 'ZCYX',},
                compression='zlib',
                bigtiff=use_bigtiff
                )
        
        # Save bbox as csv file
        if kwargs['crop_func'] is not None:
            zmin, ymin, xmin, zmax, ymax, xmax = bbox
            pd.DataFrame(
                data={
                    'zmin': [zmin],
                    'ymin': [ymin],
                    'xmin': [xmin],
                    'zmax': [zmax],
                    'ymax': [ymax],
                    'xmax': [xmax],
                    }
                ).to_csv(
                    output_path.joinpath(filename.stem + '_bbox.csv'),
                    index=False
                    )
                

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('-i','--input_path', type=str, default='../data/GBM_SORA/raw',
        help='Define input path with preprocessed images')
    parser.add_argument('-o', '--output_path', type=str, default='../data/GBM_SORA/preprocessed',
        help='Define output path where the preprocessed images will be stored')
    parser.add_argument('--output_path_binary', type=str, default=None,
                        help='Define output path where the binary images will be stored' \
                            'If None, no binary images will be stored')
    parser.add_argument('--file_extension', type=str, default='.tif',
        help='Define the file extension of the input files')
    parser.add_argument('--current_voxelsize', nargs=3, type=float, default=[1.0, 0.325, 0.325],
        help='Define the current voxel size of the raw images')
    parser.add_argument('--target_voxelsize', nargs=3, type=float, default=[1.0, 1.0, 1.0],
        help='Define the voxel size that the preprocessed images will have')
    parser.add_argument('--channels2store', nargs='+', type=int, default=0,
        help='Define which channels of the input images should be stored in the preprocessed output')
    parser.add_argument('--channels2normalize', nargs='+', type=int, default=None,
        help='Define which channels of the input images should be normalized. Default is all channels')
    parser.add_argument('--pmins', nargs='+', type=float, default=0.1,
        help='Define pmins for per channel normalization')
    parser.add_argument('--pmaxs', nargs='+', type=float, default=99.9,
        help='Define pmaxs for per channel normalization')
    parser.add_argument('--crop_func', type=types.FunctionType, default=None,
        help='Define custom crop function will be executed on the images before normalization')
    parser.add_argument('--crop_channel', type=int, default=0,
        help='Define which channel should be used for cropping')
    parser.add_argument('--masked_patch', action='store_true',
        help='Define if the patch should be masked foreground within the bounding box')
    parser.add_argument('--min_size', type=int, default=0,
        help='Define minimum size foreground objects need to have to be retained. ' \
            'If min_size a float, it is interpreted as a fraction of the total number of voxels.')
    parser.add_argument('--padding', type=int, default=0,
        help='Define custom padding which be added on the images during normalization')
    parser.add_argument('--verbose', action='store_true',
        help='Define if verbose output should be printed')
    args, _ = parser.parse_known_args()

    if isinstance(args.channels2store, int):
        args.channels2store = [args.channels2store]
    
    if isinstance(args.channels2normalize, int):
        args.channels2normalize = [args.channels2normalize]
    
    if isinstance(args.pmins, float):
        args.pmins = [args.pmins]
    
    if isinstance(args.pmaxs, float):
        args.pmaxs = [args.pmaxs]
    
    if len(args.pmins) != 1 and (len(args.pmins) != len(args.channels2store)):
        raise ValueError((
            'Number of elements in pmins should be 1 or equal '
            f'to the number of channels to store, but it is {len(args.pmins)}'
            ))

    if len(args.pmaxs) != 1 and (len(args.pmaxs) != len(args.channels2store)):
        raise ValueError((
            'Number of elements in pmaxs should be 1 or equal '
            f'to the number of channels to store, but it is {len(args.pmaxs)}'
            ))

    return args

if __name__ == '__main__':
    args = parse_args()
    main(**vars(args))