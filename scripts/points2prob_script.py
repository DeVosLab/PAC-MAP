from argparse import ArgumentParser
import os
from dotenv import load_dotenv
from pathlib import Path
from importlib.machinery import SourceFileLoader

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from pacmap import points2prob


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

    input_path = Path(data_path).joinpath(config['points2prob'][kwargs['config_key']]['input_path'])
    output_path = Path(data_path).joinpath(config['points2prob'][kwargs['config_key']]['output_path'])
    output_path.mkdir(parents=True, exist_ok=True)

    args = points2prob.parse_args()
    args.input_path = input_path
    args.output_path = output_path
    args.img_shape = config['points2prob'][kwargs['config_key']]['img_shape'] if \
        'img_shape' in config['points2prob'][kwargs['config_key']] else None
    args.img_path = Path(data_path).joinpath(config['points2prob'][kwargs['config_key']]['img_path']) if \
        'img_path' in config['points2prob'][kwargs['config_key']] else None
    args.radi_um = config['points2prob'][kwargs['config_key']]['radi_um']
    args.intensity_as_spacing = config['points2prob'][kwargs['config_key']]['intensity_as_spacing'] if \
        'intensity_as_spacing' in config['points2prob'][kwargs['config_key']] else False
    args.voxelsize = config['points2prob'][kwargs['config_key']]['voxelsize']
    args.method = config['points2prob'][kwargs['config_key']]['method']
    args.check_batches = config['points2prob'][kwargs['config_key']]['check_batches']
    args.save_points_csv = config['points2prob'][kwargs['config_key']]['save_points_csv'] if \
        'save_points_csv' in config['points2prob'][kwargs['config_key']] else False
    args.save_targets = config['points2prob'][kwargs['config_key']]['save_targets'] if \
        'save_targets' in config['points2prob'][kwargs['config_key']] else False

    points2prob.main(**vars(args))

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-f', '--dataset_config_file', type=str, required=True,
                        help='Path to dataset configuration file')
    parser.add_argument('-k', '--config_key', type=str, default='targets',
                        help='Key of dataset configuration to use')
    args = parser.parse_args()
    main(**vars(args))