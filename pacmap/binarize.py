from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm
import tifffile
import numpy as np
from skimage.filters import threshold_triangle
from skimage.morphology import binary_opening, remove_small_objects
from scipy.ndimage import binary_fill_holes, median_filter


def main(**kwargs):
    assert kwargs['input_path'] is not None, 'Define input path'
    assert kwargs['output_path'] is not None, 'Define output path'

    input_path = Path(kwargs['input_path'])
    output_path = Path(kwargs['output_path'])
    if ~output_path.is_dir():
        output_path.mkdir(parents=True, exist_ok=True)

    # Get all filenames in input path
    filenames = [f for f in input_path.iterdir() if \
        f.is_file() and f.suffix == kwargs['file_extension']]

    # Loop over all files
    for filename in tqdm(filenames):
        img = tifffile.imread(filename)
        if img.ndim == 4: # Select channel to use
            img = img[:,kwargs['channel2use'],]
        if img.ndim == 3: # Add channel dimension if necessary
            channel_dim=1
            img = np.expand_dims(img, channel_dim)

        # Binarize image, fill holes, remove small objects and apply median filter
        foreground = img > threshold_triangle(img)
        foreground = binary_opening(foreground, np.ones((9,1,9,9)))
        foreground = binary_fill_holes(foreground)
        foreground = remove_small_objects(foreground, min_size=1000)
        foreground = median_filter(foreground, size=(1, 1, 5, 5)).astype(bool)

        # Save binarized image
        tifffile.imwrite(
            output_path.joinpath(filename.name),
            foreground.astype(np.float32),
            imagej=True,
            metadata={
                'axes': 'ZCYX',
                },
            compression='zlib'
            )

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('-i','--input_path', type=str, default=None,
        help='Define input path with preprocessed images')
    parser.add_argument('-o', '--output_path', type=str, default=None,
        help='Define output path where the preprocessed images will be stored')
    parser.add_argument('--file_extension', type=str, default='.tif',
        help='Define the file extension of the input files')
    parser.add_argument('--channel2use', type=int, default=0,
        help='Define which channel of the input image should be used for binarization')
    args, _ = parser.parse_known_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    main(**vars(args))