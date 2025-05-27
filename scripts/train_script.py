from argparse import ArgumentParser
import os
from dotenv import load_dotenv
from pathlib import Path
from importlib.machinery import SourceFileLoader

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from pacmap import train


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

    # Train model for each random seed
    random_seeds = config['train'][kwargs['config_key']]['random_seed']
    if isinstance(random_seeds, int):
        random_seeds = [random_seeds]
    
    for i, random_seed in enumerate(random_seeds):
        print(f'Random seed: {random_seed} (iteration {i+1}/{len(random_seeds)})')

        args = train.parse_args()

        args.input_path = Path(data_path).joinpath(config['train'][kwargs['config_key']]['input_path'])
        args.binary_path = Path(data_path).joinpath(config['train'][kwargs['config_key']]['binary_path']) if \
            'binary_path' in config['train'][kwargs['config_key']] else None
        args.target_path = Path(data_path).joinpath(config['train'][kwargs['config_key']]['target_path'])
        args.output_path = Path(data_path).joinpath(config['train'][kwargs['config_key']]['output_path'])
        args.normalize_targets = config['train'][kwargs['config_key']]['normalize_targets'] if \
            'normalize_targets' in config['train'][kwargs['config_key']] else True
        args.split = config['train'][kwargs['config_key']]['split']
        args.random_seed = random_seed
        args.min_percentage = config['train'][kwargs['config_key']]['min_percentage'] if \
            'min_percentage' in config['train'][kwargs['config_key']] else None
        args.batch_size = config['train'][kwargs['config_key']]['batch_size']
        args.num_epochs = config['train'][kwargs['config_key']]['num_epochs']
        args.lr = config['train'][kwargs['config_key']]['lr']
        args.model_type = config['train'][kwargs['config_key']]['model_type'] if \
            'model_type' in config['train'][kwargs['config_key']] else 'UNet3D'
        args.pretrained = Path(data_path).joinpath(config['train'][kwargs['config_key']]['pretrained']) if \
            'pretrained' in config['train'][kwargs['config_key']] else None
        args.final_sigmoid = config['train'][kwargs['config_key']]['final_sigmoid'] if \
            'final_sigmoid' in config['train'][kwargs['config_key']] else False
        args.loss = config['train'][kwargs['config_key']]['loss'] if \
            'loss' in config['train'][kwargs['config_key']] else 'MSE'
        args.patience = config['train'][kwargs['config_key']]['patience'] if \
            'patience' in config['train'][kwargs['config_key']] else 5
        args.f_maps = config['train'][kwargs['config_key']]['f_maps']
        args.depth = config['train'][kwargs['config_key']]['depth'] if \
            'depth' in config['train'][kwargs['config_key']] else 4
        args.gpu_id = config['train'][kwargs['config_key']]['gpu_id']
        args.test_only = config['train'][kwargs['config_key']]['test_only'] if \
            'test_only' in config['train'][kwargs['config_key']] else False
        args.augment_rescale_p = config['train'][kwargs['config_key']]['augment_rescale_p'] if \
            'augment_rescale_p' in config['train'][kwargs['config_key']] else 0.5
        args.augment_rescale_range = config['train'][kwargs['config_key']]['augment_rescale_range'] if \
            'augment_rescale_range' in config['train'][kwargs['config_key']] else [0.75, 1.25]
        args.augment_rescale_anisotropic = config['train'][kwargs['config_key']]['augment_rescale_anisotropic'] if \
            'augment_rescale_anisotropic' in config['train'][kwargs['config_key']] else False
        args.augment_brightness_sigma = config['train'][kwargs['config_key']]['augment_brightness_sigma'] if \
            'augment_brightness_sigma' in config['train'][kwargs['config_key']] else 0.1
        args.augment_contrast_range = config['train'][kwargs['config_key']]['augment_contrast_range'] if \
            'augment_contrast_range' in config['train'][kwargs['config_key']] else [0.9, 1.1]
        args.augment_noise_p = config['train'][kwargs['config_key']]['augment_noise_p'] if \
            'augment_noise_p' in config['train'][kwargs['config_key']] else 0.5
        args.detect_anomaly = config['train'][kwargs['config_key']]['detect_anomaly'] if \
            'detect_anomaly' in config['train'][kwargs['config_key']] else False
        args.check_batches = config['train'][kwargs['config_key']]['check_batches'] if \
            'check_batches' in config['train'][kwargs['config_key']] else True

        train.main(**vars(args))

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-f', '--dataset_config_file', type=str, required=True,
                        help='Path to dataset configuration file')
    parser.add_argument('-k', '--config_key', type=str, default='scratch',
                        help='Key of dataset configuration to use')
    args = parser.parse_args()
    main(**vars(args))