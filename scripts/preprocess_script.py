from argparse import ArgumentParser
import os
from dotenv import load_dotenv
from pathlib import Path
from importlib.machinery import SourceFileLoader

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from pacmap import preprocess


def main(**kwargs):
    # Load environment variables
    load_dotenv()
    data_path = os.getenv('DATA_PATH')

    # Load dataset configuration stored as dict in python file
    config_file = Path(kwargs['dataset_config_file'])
    config = SourceFileLoader(config_file.name, str(config_file)).load_module().config

    input_path = Path(data_path).joinpath(config['preprocess']['input_path'])
    output_path = Path(data_path).joinpath(config['preprocess']['output_path'])
    output_path.mkdir(parents=True, exist_ok=True)
    output_path_binary = Path(data_path).joinpath(config['preprocess']['output_path_binary']) if \
        'output_path_binary' in config['preprocess'] else None
    if output_path_binary is not None:
        output_path_binary.mkdir(parents=True, exist_ok=True)

    args = preprocess.parse_args()
    args.input_path = input_path
    args.output_path = output_path
    args.output_path_binary = output_path_binary
    args.channels2store = config['preprocess']['channels2store']
    args.channels2normalize = config['preprocess']['channels2normalize'] if 'channels2normalize' in config['preprocess'] else None
    args.current_voxelsize = config['preprocess']['current_voxelsize']
    args.target_voxelsize = config['preprocess']['target_voxelsize']
    args.crop_func = config['preprocess']['crop_func'] if 'crop_func' in config['preprocess'] else None
    args.crop_channel = config['preprocess']['crop_channel'] if 'crop_channel' in config['preprocess'] else None
    args.masked_patch = config['preprocess']['masked_patch'] if 'masked_patch' in config['preprocess'] else False
    args.min_size = config['preprocess']['min_size'] if 'min_size' in config['preprocess'] else None
    args.pmins = config['preprocess']['pmins'] if 'pmins' in config['preprocess'] else args.pmins # overwrite default
    args.pmaxs = config['preprocess']['pmaxs'] if 'pmaxs' in config['preprocess'] else args.pmaxs # overwrite default

    preprocess.main(**vars(args))

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('-f', '--dataset_config_file', type=str, required=True,
                        help='Path to dataset configuration file')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    main(**vars(args))