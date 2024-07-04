from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm
import tifffile
import numpy as np
from patchify import patchify

from .utils import get_padding


def main(**kwargs):
    input_path = Path(kwargs['input_path'])
    output_path = Path(kwargs['output_path'])
    if ~output_path.is_dir():
        output_path.mkdir(parents=True, exist_ok=True)

    # Insert element for channel dimension to patch_size and patch_stride
    channel_dim = kwargs['channel_dim']
    n_channels = len(kwargs['channels2store'])

    patch_size = list(kwargs['patch_size'])
    patch_stride = list(kwargs['patch_stride'])
    patch_size.insert(channel_dim, n_channels)
    patch_stride.insert(channel_dim, n_channels) 
    patch_size = tuple(patch_size)
    patch_stride = tuple(patch_stride)

    # Get all filenames in input path
    filenames = [f for f in input_path.iterdir() if \
        f.is_file() and f.suffix == kwargs['file_extension'] and \
        not f.stem.startswith('.')]
    
    # Loop over all files
    for i, filename in enumerate(filenames):
        if kwargs['verbose']:
            print(f'Sample {i+1}/{len(filenames)}')
        img = tifffile.imread(filename)
        if img.ndim == 4:
            img = img[:,kwargs['channels2store'],]
        if img.ndim == 3:
            img = np.expand_dims(img, channel_dim)
        
        # Define target size and padding
        target_size = patch_size*np.ceil([a/b for (a,b) in zip(img.shape, patch_size)]).astype(int)
        if kwargs['verbose']:
            print(f'Target size: {target_size}')
            print('Padding image to target size')
        padding = get_padding(img, shape=target_size, multichannel=True)
        img = np.pad(img, padding, mode='constant')

        # Create patches
        if kwargs['verbose']:
            print('Creating patches')
        patches = patchify(img, patch_size, patch_stride)
        patches_structure = patches.shape[:4]
        n_patches = np.prod(patches_structure)
        if kwargs['verbose']:
            print(f'Number of patches: {n_patches}')
            print(f'Patch structure: {patches_structure}')
            print('Flattening patch structure')
        patches = patches.reshape((n_patches,) + patch_size)
        
        # Define metadata
        metadata = {
            'axes': 'ZCYX',
            'patch_size': patch_size,
            'patch_stride': patch_stride,
            'patches_structure': patches_structure,
            'original_img_shape': img.shape
        }
        if kwargs['verbose']:
            print(f'Metadata: {metadata}')

        # Save patches
        if kwargs['verbose']:
            print('Saving patches')
        for p in tqdm(range(n_patches)):
            sample_name = filename.stem
            if kwargs['store_batches']:
                output_path_sample = output_path.joinpath(sample_name)
                output_path_sample.mkdir(exist_ok=True)
            else:
                output_path_sample = output_path
            patch = patches[p,]
            tifffile.imwrite(
                output_path_sample.joinpath(f'{sample_name}_patch_{p:04}.tif'),
                patch.astype(np.float32),
                imagej=True,
                metadata=metadata,
                compression='zlib'
                )

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('-i','--input_path', type=str, default='../data/GBM_SORA/preprocessed',
        help='Define input path with preprocessed images')
    parser.add_argument('-o', '--output_path', type=str, default='../data/GBM_SORA/patches',
        help='Define output path where the patches will be stored')
    parser.add_argument('--file_extension', type=str, default='.tif',
        help='Define the file extension of the input files')
    parser.add_argument('--channel_dim', type=int, default=1,
        help='Define the channel dimension of the input images')
    parser.add_argument('--channels2store', nargs='+', type=int, default=0,
        help='Define which channels of the input images should be stored in the patches')
    parser.add_argument('--patch_size', nargs=3, type=int, default=[128, 128, 128],
        help='Define the size of the patches in Z, Y and X dim')
    parser.add_argument('--patch_stride', nargs=3, type=int, default=[64, 64, 64],
        help='Define the stride between created patches in Z, Y and X dim')
    parser.add_argument('--store_batches', type=bool, default=True,
        help='Store patches in batch subfolders (samples) or not')
    parser.add_argument('--verbose', type=bool, action='store_true',
        help='Print more information')
    args, _ = parser.parse_known_args()

    if isinstance(args.channels2store, int):
        args.channels2store = [args.channels2store]

    return args

if __name__ == '__main__':
    args = parse_args()
    main(**vars(args))