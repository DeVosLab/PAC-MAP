from argparse import ArgumentParser
import os
from dotenv import load_dotenv
from pathlib import Path
from importlib.machinery import SourceFileLoader

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from pacmap import patch_creation


def main(**kwargs):
    # Load environment variables
    load_dotenv()
    data_path = os.getenv('DATA_PATH')

    # Load dataset configuration stored as dict in python file
    config_file = Path(kwargs['dataset_config_file'])
    config = SourceFileLoader(config_file.name, str(config_file)).load_module().config

    input_path = Path(data_path).joinpath(config['patch_creation'][kwargs['config_key']]['input_path'])
    output_path = Path(data_path).joinpath(config['patch_creation'][kwargs['config_key']]['output_path'])
    output_path.mkdir(parents=True, exist_ok=True)

    args = patch_creation.parse_args()

    args.input_path = input_path
    args.output_path = output_path
    args.channels2store = config['patch_creation'][kwargs['config_key']]['channels2store']
    args.patch_size = config['patch_creation'][kwargs['config_key']]['patch_size']
    args.patch_stride = config['patch_creation'][kwargs['config_key']]['patch_stride']
    args.store_batches = config['patch_creation'][kwargs['config_key']]['store_batches'] if \
        'store_batches' in config['patch_creation'][kwargs['config_key']] else True

    patch_creation.main(**vars(args))

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-f', '--dataset_config_file', type=str, required=True,
                        help='Path to dataset configuration file')
    parser.add_argument('-k', '--config_key', type=str, required=True,
                        help='Key of dataset configuration to use')
    args = parser.parse_args()
    main(**vars(args))