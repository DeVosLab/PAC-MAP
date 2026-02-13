from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm
import numpy as np
import tifffile
from scipy.ndimage import binary_fill_holes

def main(args):
    mask_path = Path(args.mask_path)
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    mask_files = sorted([mask for mask in mask_path.glob('*.tif') if not mask.name.startswith('.')])
    for mask_file in tqdm(mask_files):
        mask = tifffile.imread(mask_file).astype(bool)
        if mask.ndim == 4:
            mask = np.squeeze(mask,axis=1)
        # 3D fill holes
        mask = binary_fill_holes(mask)
        # 2D fill holes
        for z in range(mask.shape[0]):
            mask[z] = binary_fill_holes(mask[z])

        output_file = output_path.joinpath(mask_file.name)
        tifffile.imwrite(output_file, mask.astype(np.uint8), imagej=True, compression='zlib', metadata={'axes': 'ZYX'})

if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument('--mask_path', type=str, required=True)
    args.add_argument('--output_path', type=str, required=True)
    args = args.parse_args()
    main(args)