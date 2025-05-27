from argparse import ArgumentParser
import os
from dotenv import load_dotenv
from pathlib import Path
from importlib.machinery import SourceFileLoader

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from pacmap import compare2ground_truth


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

    args = compare2ground_truth.parse_args()
    args.filename = Path(data_path).joinpath(config['compare2ground_truth'][kwargs['config_key']]['filename'])
    args.output_path = Path(data_path).joinpath(config['compare2ground_truth'][kwargs['config_key']]['output_path'])
    args.gt_name = config['compare2ground_truth'][kwargs['config_key']]['gt_name']
    args.metric = config['compare2ground_truth'][kwargs['config_key']]['metric']
    args.voxelsize = config['compare2ground_truth'][kwargs['config_key']]['voxelsize']
    args.max_dist_um = config['compare2ground_truth'][kwargs['config_key']]['max_dist_um']
    args.palette = config['compare2ground_truth'][kwargs['config_key']]['palette'] if \
        'palette' in config['compare2ground_truth'][kwargs['config_key']] else None
    args.stats = config['compare2ground_truth'][kwargs['config_key']]['stats'] if \
        'stats' in config['compare2ground_truth'][kwargs['config_key']] else False
    compare2ground_truth.main(**vars(args))

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('-f', '--dataset_config_file', type=str, required=True,
                        help='Path to dataset configuration file')
    parser.add_argument('-k', '--config_key', type=str, default='weak_targets',
                        help='Key of dataset configuration to use')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    main(**vars(args))