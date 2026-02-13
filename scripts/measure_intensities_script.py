from argparse import ArgumentParser
from argparse import Namespace
import os
from dotenv import load_dotenv
from pathlib import Path
from importlib.machinery import SourceFileLoader

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from pacmap import measure_intensities


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

    args = Namespace()
    cfg = config['measure_intensities'][kwargs['config_key']]

    args.depatchified_path = Path(data_path).joinpath(cfg['depatchified_path'])
    args.output_path = Path(data_path).joinpath(cfg['output_path'])
    args.voxelsize = cfg['voxelsize']
    args.radius_um = cfg['radius_um']
    args.check_batches = cfg['check_batches'] if 'check_batches' in cfg else True

    measure_intensities.main(**vars(args))


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('-f', '--dataset_config_file', type=str, required=True,
                        help='Path to dataset configuration file')
    parser.add_argument('-k', '--config_key', type=str, default='default',
                        help='Key of dataset configuration to use')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    main(**vars(args))
