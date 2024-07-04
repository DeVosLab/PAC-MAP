from argparse import ArgumentParser
import os
from dotenv import load_dotenv
from pathlib import Path
from importlib.machinery import SourceFileLoader

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from pacmap import weak_targets


def main(**kwargs):
    # Load environment variables
    load_dotenv()
    data_path = os.getenv('DATA_PATH')

    # Load dataset configuration stored as dict in python file
    config_file = Path(kwargs['dataset_config_file'])
    config = SourceFileLoader(config_file.name, str(config_file)).load_module().config

    input_path_grayscale = Path(data_path).joinpath(config['weak_targets']['input_path_grayscale'])
    input_path_binary = Path(data_path).joinpath(config['weak_targets']['input_path_binary'])
    output_path = Path(data_path).joinpath(config['weak_targets']['output_path'])
    output_path.mkdir(parents=True, exist_ok=True)

    args = weak_targets.parse_args()

    args.input_path_grayscale = input_path_grayscale
    args.input_path_binary = input_path_binary
    args.output_path = output_path
    args.radi_um = config['weak_targets']['radi_um']
    args.voxelsize = config['weak_targets']['voxelsize']
    args.min_distance = config['weak_targets']['min_distance']
    args.exclude_border = config['weak_targets']['exclude_border']
    args.save_binary = config['weak_targets']['save_binary']
    args.save_labels = config['weak_targets']['save_labels']
    args.save_points_npy = config['weak_targets']['save_points_npy']
    args.save_points_csv = config['weak_targets']['save_points_csv']
    args.save_points_img = config['weak_targets']['save_points_img']
    args.save_targets = config['weak_targets']['save_targets']

    weak_targets.main(**vars(args))

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-f', '--dataset_config_file', type=str, required=True,
                        help='Path to dataset configuration file')
    args = parser.parse_args()
    main(**vars(args))