from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import tifffile

def main(args):
    img_path = Path(args.img_path)
    bbox_path = Path(args.bbox_path)
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    img_files = sorted([img for img in img_path.glob('*.tif') if not img.name.startswith('.')])
    for img_file in tqdm(img_files):
        img = tifffile.imread(img_file)
        bbox_file = bbox_path.joinpath(img_file.stem + '_bbox.csv')
        bbox = pd.read_csv(bbox_file)
        img = img[
            bbox['zmin'][0]:bbox['zmax'][0],
            :,
            bbox['ymin'][0]:bbox['ymax'][0],
            bbox['xmin'][0]:bbox['xmax'][0]
            ]
        output_file = output_path.joinpath(img_file.name)
        tifffile.imwrite(output_file, img, imagej=True, compression='zlib')

if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument('--img_path', type=str, required=True)
    args.add_argument('--bbox_path', type=str, required=True)
    args.add_argument('--output_path', type=str, required=True)
    args = args.parse_args()
    main(args)