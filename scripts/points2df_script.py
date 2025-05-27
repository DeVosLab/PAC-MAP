from argparse import ArgumentParser
import os
from dotenv import load_dotenv
from pathlib import Path
from importlib.machinery import SourceFileLoader

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from pacmap import points2df


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

    args = points2df.parse_args()

    args.dataset_names = config['points2df'][kwargs['config_key']]['dataset_names']
    input_paths = [Path(data_path).joinpath(input_path) for \
                   input_path in config['points2df'][kwargs['config_key']]['input_paths']]
    args.input_paths = input_paths
    args.methods = config['points2df'][kwargs['config_key']]['methods']
    args.output_path = Path(data_path).joinpath(config['points2df'][kwargs['config_key']]['output_path'])
    args.voxelsize = config['points2df'][kwargs['config_key']]['voxelsize']
    args.volumesize_vox = config['points2df'][kwargs['config_key']]['volumesize_vox'] if \
        'volumesize_vox' in config['points2df'][kwargs['config_key']] else None
    args.bordersize_um = config['points2df'][kwargs['config_key']]['bordersize_um'] if \
        'bordersize_um' in config['points2df'][kwargs['config_key']] else None
    args.check_batches = config['points2df'][kwargs['config_key']]['check_batches'] if \
        'check_batches' in config['points2df'][kwargs['config_key']] else True
    args.merge_distance_um = config['points2df'][kwargs['config_key']]['merge_distance_um'] if \
        'merge_distance_um' in config['points2df'][kwargs['config_key']] else -1

    points2df.main(**vars(args))

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-f', '--dataset_config_file', type=str, required=True,
                        help='Path to dataset configuration file')
    parser.add_argument('-k', '--config_key', type=str, default='weak_targets',
                        help='Key of dataset configuration to use')
    args = parser.parse_args()
    main(**vars(args))