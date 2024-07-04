from argparse import ArgumentParser
import os
from dotenv import load_dotenv
from pathlib import Path
from importlib.machinery import SourceFileLoader

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from pacmap import binarize


def main(**kwargs):
    # Load environment variables
    load_dotenv()
    data_path = os.getenv('DATA_PATH')

    # Load dataset configuration stored as dict in python file
    config_file = Path(kwargs['dataset_config_file'])
    config = SourceFileLoader(config_file.name, str(config_file)).load_module().config

    input_path = Path(data_path).joinpath(config['binarize']['input_path'])
    output_path = Path(data_path).joinpath(config['binarize']['output_path'])
    output_path.mkdir(parents=True, exist_ok=True)

    args = binarize.parse_args()

    args.input_path = input_path
    args.output_path = output_path
    args.channel2use = config['binarize']['channel2use']

    binarize.main(**vars(args))

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-f', '--dataset_config_file', type=str, required=True,
                        help='Path to dataset configuration file')
    args = parser.parse_args()
    main(**vars(args))