from argparse import ArgumentParser
import os
from dotenv import load_dotenv
from pathlib import Path
from importlib.machinery import SourceFileLoader

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from pacmap import prob2points


def main(**kwargs):
    # Load environment variables
    load_dotenv()
    data_path = os.getenv('DATA_PATH')

    # Load dataset configuration stored as dict in python file
    config_file = Path(kwargs['dataset_config_file'])
    config = SourceFileLoader(config_file.name, str(config_file)).load_module().config

    input_path_probs = Path(data_path).joinpath(config['prob2points'][kwargs['config_key']]['input_path_probs'])
    input_path_masks = Path(data_path).joinpath(config['prob2points'][kwargs['config_key']]['input_path_masks'])
    output_path = Path(data_path).joinpath(config['prob2points'][kwargs['config_key']]['output_path'])
    output_path.mkdir(parents=True, exist_ok=True)

    args = prob2points.parse_args()

    args.input_path_probs = input_path_probs
    args.input_path_masks = input_path_masks
    args.output_path = output_path
    args.radi_um = config['prob2points'][kwargs['config_key']]['radi_um']
    args.voxelsize = config['prob2points'][kwargs['config_key']]['voxelsize']
    args.min_distance = config['prob2points'][kwargs['config_key']]['min_distance']
    args.threshold_abs = config['prob2points'][kwargs['config_key']]['threshold_abs'] if \
        'threshold_abs' in config['prob2points'][kwargs['config_key']] else None
    args.exclude_border = config['prob2points'][kwargs['config_key']]['exclude_border'] if \
        'exclude_border' in config['prob2points'][kwargs['config_key']] else False
    args.save_points_csv = config['prob2points'][kwargs['config_key']]['save_points_csv']
    args.intensity_as_spacing = config['points2prob'][kwargs['config_key']]['intensity_as_spacing'] if \
        'intensity_as_spacing' in config['points2prob'][kwargs['config_key']] else False
    args.top_down = config['points2prob'][kwargs['config_key']]['top_down'] if \
        'top_down' in config['points2prob'][kwargs['config_key']] else True
    args.merge_close_points = config['points2prob'][kwargs['config_key']]['merge_close_points'] if \
        'merge_close_points' in config['points2prob'][kwargs['config_key']] else True

    prob2points.main(**vars(args))

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-f', '--dataset_config_file', type=str, required=True,
                        help='Path to dataset configuration file')
    parser.add_argument('-k', '--config_key', type=str, default='weak_targets',
                        help='Key of dataset configuration to use')
    args = parser.parse_args()
    main(**vars(args))