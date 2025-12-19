from argparse import ArgumentParser
import os
from dotenv import load_dotenv
from pathlib import Path
from importlib.machinery import SourceFileLoader

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from pacmap import depatchify


def main(**kwargs):
    # Load dataset configuration stored as dict in python file
    config_file = Path(kwargs['dataset_config_file'])
    config = SourceFileLoader(config_file.name, str(config_file)).load_module().config

    # Get the data path
    data_path = config.get('DATA_PATH', None)
    if data_path is None:
        load_dotenv()
        data_path = os.getenv('DATA_PATH')
    if data_path is None:
        raise ValueError('DATA_PATH is not set in the config file or environment variables')


    args = depatchify.parse_args()

    args.input_path = Path(data_path).joinpath(config['depatchify']['input_path'])
    args.binary_path = Path(data_path).joinpath(config['depatchify']['binary_path'])
    args.preds_path = Path(data_path).joinpath(config['depatchify']['preds_path'])
    args.coords_path = Path(data_path).joinpath(config['depatchify']['coords_path'])
    args.output_path = Path(data_path).joinpath(config['depatchify']['output_path'])
    args.do_fill_holes = config['depatchify']['do_fill_holes']
    args.do_mask_points = config['depatchify']['do_mask_points']
    args.remove_zero_padding = config['depatchify']['remove_zero_padding']
    args.voxelsize = config['raw']['voxelsize']
    depatchify.main(**vars(args))

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-f', '--dataset_config_file', type=str, required=True,
                        help='Path to dataset configuration file')
    args = parser.parse_args()
    main(**vars(args))