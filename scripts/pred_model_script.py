from argparse import ArgumentParser
import os
from dotenv import load_dotenv
from pathlib import Path
from importlib.machinery import SourceFileLoader

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from pacmap import pred_model


def main(**kwargs):
    # Load environment variables
    load_dotenv()
    data_path = os.getenv('DATA_PATH')

    # Load dataset configuration stored as dict in python file
    config_file = Path(kwargs['dataset_config_file'])
    config = SourceFileLoader(config_file.name, str(config_file)).load_module().config

    args = pred_model.parse_args()
    args.input_path = Path(data_path).joinpath(config['pred_model'][kwargs['config_key']]['input_path'])
    args.output_path = Path(data_path).joinpath(config['pred_model'][kwargs['config_key']]['output_path'])
    args.model_path = Path(data_path).joinpath(config['pred_model'][kwargs['config_key']]['model_path'])
    args.model_type = config['pred_model'][kwargs['config_key']]['model_type'] if \
        'model_type' in config['pred_model'][kwargs['config_key']] else 'UNet3D'
    args.final_sigmoid = config['pred_model'][kwargs['config_key']]['final_sigmoid'] if \
        'final_sigmoid' in config['pred_model'][kwargs['config_key']] else False
    args.f_maps = config['pred_model'][kwargs['config_key']]['f_maps']
    args.depth = config['pred_model'][kwargs['config_key']]['depth'] if \
        'depth' in config['pred_model'][kwargs['config_key']] else 4
    args.gpu_id = config['pred_model'][kwargs['config_key']]['gpu_id']
    args.min_distance = config['pred_model'][kwargs['config_key']]['min_distance'] if \
        'min_distance' in config['pred_model'][kwargs['config_key']] else None
    args.merge_close_points = config['pred_model'][kwargs['config_key']]['merge_close_points'] if \
        'merge_close_points' in config['pred_model'][kwargs['config_key']] else True
    args.do_border = config['pred_model'][kwargs['config_key']]['do_border'] if \
        'do_border' in config['pred_model'][kwargs['config_key']] else False
    args.threshold_abs = config['pred_model'][kwargs['config_key']]['threshold_abs'] if \
        'threshold_abs' in config['pred_model'][kwargs['config_key']] else None
    args.intensity_as_spacing = config['pred_model'][kwargs['config_key']]['intensity_as_spacing'] if \
        'intensity_as_spacing' in config['pred_model'][kwargs['config_key']] else False
    args.top_down = config['pred_model'][kwargs['config_key']]['top_down'] if \
        'top_down' in config['pred_model'][kwargs['config_key']] else True
    args.voxelsize = config['pred_model'][kwargs['config_key']]['voxelsize']
    args.save_csv = config['pred_model'][kwargs['config_key']]['save_csv'] if \
        'save_csv' in config['pred_model'][kwargs['config_key']] else False
    args.save_preds = config['pred_model'][kwargs['config_key']]['save_preds'] if \
        'save_preds' in config['pred_model'][kwargs['config_key']] else False
    args.channel2use = config['pred_model'][kwargs['config_key']]['channel2use'] if \
        'channel2use' in config['pred_model'][kwargs['config_key']] else None
    args.check_batches = config['pred_model'][kwargs['config_key']]['check_batches'] if \
        'check_batches' in config['pred_model'][kwargs['config_key']] else True

    pred_model.main(**vars(args))

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-f', '--dataset_config_file', type=str, required=True,
                        help='Path to dataset configuration file')
    parser.add_argument('-k', '--config_key', type=str, default='weakly_supervised',
                        help='Key of dataset configuration to use')
    args = parser.parse_args()
    main(**vars(args))