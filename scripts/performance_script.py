from argparse import ArgumentParser
import os
from dotenv import load_dotenv
from pathlib import Path
from importlib.machinery import SourceFileLoader

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from pacmap import performance


def main(**kwargs):
    # Load environment variables
    load_dotenv()
    data_path = Path(os.getenv('DATA_PATH'))

    # Load dataset configuration stored as dict in python file
    config_file = Path(kwargs['dataset_config_file'])
    config = SourceFileLoader(config_file.name, str(config_file)).load_module().config

    args = performance.parse_args()

    args.true_path = data_path.joinpath(config['performance'][kwargs['config_key']]['true_path'])
    args.points_path = data_path.joinpath(config['performance'][kwargs['config_key']]['points_path'])
    args.output_path = data_path.joinpath(config['performance'][kwargs['config_key']]['output_path'])
    args.filename = config['performance'][kwargs['config_key']]['filename']
    args.score_method = config['performance'][kwargs['config_key']]['score_method'] if \
        'score_method' in config['performance'][kwargs['config_key']] else 'own'
    args.true2points_method = config['performance'][kwargs['config_key']]['true2points_method']
    args.pred2points_method = config['performance'][kwargs['config_key']]['pred2points_method']
    args.voxelsize = config['performance'][kwargs['config_key']]['voxelsize']
    args.volumesize = config['performance'][kwargs['config_key']]['volumesize']
    args.bordersize_um = config['performance'][kwargs['config_key']]['bordersize_um']
    args.threshold = config['performance'][kwargs['config_key']]['threshold']
    args.intensity_as_spacing = config['performance'][kwargs['config_key']]['intensity_as_spacing'] if \
        'intensity_as_spacing' in config['performance'][kwargs['config_key']] else False
    args.top_down = config['performance'][kwargs['config_key']]['top_down'] if \
        'top_down' in config['performance'][kwargs['config_key']] else True

    performance.main(**vars(args))

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-f', '--dataset_config_file', type=str, required=True,
                        help='Path to dataset configuration file')
    parser.add_argument('-k', '--config_key', type=str, default='weak_targets',
                        help='Key of dataset configuration to use')
    args = parser.parse_args()
    main(**vars(args))