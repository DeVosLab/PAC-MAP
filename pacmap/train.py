import sys
from argparse import ArgumentParser
import logging
import traceback
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import json
import pandas as pd
import tifffile
import numpy as np
import numbers
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose
from torch.utils.tensorboard import SummaryWriter
from pytorch_msssim import ms_ssim

import importlib
pytorch3dunet = importlib.import_module('pytorch-3dunet.pytorch3dunet.unet3d.model')

from .utils import (unsqueeze_to_ndim, seed_worker, set_random_seeds,
                   RandomFlip3D, RandomRescaledCrop3D, AugmentBrightness,
                   AugmentContrast, AugmentGaussianNoise, TverskyLoss)


def get_data(input_path, binary_path=None, target_path=None, check_batches=True, suffix='.tif'):
    '''
    Creates a pandas.DataFrame with the batch and paths to the input images, binary 
    images and the targets (optional)

    Parameters
    ----------
    batch: str
        Batch name
    input_path : pathlib.Path
        Path to the input images
    binary_path : pathlib.Path or None
        Path to the binary images (optional)
    target_path : pathlib.Path or None
        Path to the target images (optional)
    check_batches : bool
        Wether to check for batches (parent directories for image files) 
        or not, by default True
    
    Returns
    -------
    pandas.DataFrame
        DataFrame with the batch and paths to the input images, binary images (optional) and the targets (optional)
    '''

    columns = ['input']
    if binary_path is not None:
        columns.append('binary')
    if target_path is not None:
        columns.append('target')
    
    input_path = Path(input_path)
    if binary_path is not None:
        binary_path = Path(binary_path)
    if target_path is not None:
        target_path = Path(target_path)

    if check_batches:
        columns.append('batch')
        batches_input = sorted([batch.name for batch in input_path.iterdir() if batch.is_dir()])
        if binary_path is not None:
            batches_binary = sorted([batch.name for batch in binary_path.iterdir() if batch.is_dir()])
        if target_path is not None:
            batches_target = sorted([batch.name for batch in target_path.iterdir() if batch.is_dir()])
    else:
        batches_input = ['']
        if binary_path is not None:
            batches_binary = ['']
        if target_path is not None:
            batches_target = ['']

    if binary_path is not None and batches_input != batches_binary:
        raise ValueError('The batches are not the same')
    if target_path is not None and batches_input != batches_target:
        raise ValueError('The batches are not the same')

    df = pd.DataFrame(columns=columns)

    for batch in batches_input:
        input_imgs = sorted([img for img in (input_path / batch).iterdir() if img.is_file() and img.suffix == suffix and not img.stem.startswith('.')])
        if binary_path is not None:
            binary_imgs = sorted([img for img in (binary_path / batch).iterdir() if img.is_file() and img.suffix == suffix and not img.stem.startswith('.')])
        if target_path is not None:
            target_imgs = sorted([img for  img in (target_path / batch).iterdir() if img.is_file() and img.suffix == suffix and not img.stem.startswith('.')])

        # Only keep overlapping images
        if binary_path is not None:
            input_imgs = [img for img in input_imgs if img.name in [img.name for img in binary_imgs]]
        if target_path is not None:
            input_imgs = [img for img in input_imgs if img.name in [img.name for img in target_imgs]]
        if binary_path is not None:
            binary_imgs = [img for img in binary_imgs if img.name in [img.name for img in input_imgs]]
        if target_path is not None:
            target_imgs = [img for img in target_imgs if img.name in [img.name for img in input_imgs]]

        # Check that the images are the same
        if binary_path is not None and [img.name for img in input_imgs] != [img.name for img in binary_imgs]:
            raise ValueError('The images are not the same')
        if target_path is not None and [img.name for img in input_imgs] != [img.name for img in target_imgs]:
            raise ValueError('The images are not the same')

        # Concatenate the input, binary and target paths to the dataframe
        n = len(input_imgs)
        entry = pd.DataFrame({
            'batch': n*[batch],
            'input': [str(img) for img in input_imgs]
        })
        if binary_path is not None:
            entry['binary'] = [str(img) for img in binary_imgs]
                
        if target_path is not None:
            entry['target'] = [str(img) for img in target_imgs]
        df = pd.concat([df,entry],axis=0,ignore_index=True)
    return df
        

def get_foreground_percentage(df):
    '''
    Loops over a dataframe with paths to the input images, binary images and targets and
    calculates the foreground percentage for each patch
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with the paths to the input images, binary images and the targets
    
    Returns
    -------
    pandas.DataFrame
        DataFrame with the paths to the input images, binary images and the targets
        with foregruond percentage for every patch
    '''
    # Check if binary paths are provided
    if 'binary' not in df.columns:
        print('Foreground percentage not calculated since binary paths are not provided')
        return df

    # Load binary images and calculate foreground percentage
    df['foreground_percentage'] = np.zeros((len(df), 0)).astype(float).tolist()
    for i, row in tqdm(df.iterrows(), colour='green', total=len(df), desc='Calculating foreground percentages'):
        binary = tifffile.imread(row['binary']).astype(bool)
        foreground_percentage = binary.sum() / binary.size
        df.loc[i, 'foreground_percentage'] = foreground_percentage
    return df

def split_dataframe(df, split, random_seed=0):
    '''
    Split a dataframe into train, validation and test according to the provided train, val and test split.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with the paths to the input images, binary images and the targets
    split: list or tuple
        List or tuple with the train, val and test split.
    random_seed : int
        Random seed for the train test split, by default 0

    Returns
    -------
    pandas.DataFrame
        DataFrame with the paths to the input images, binary images and the targets
        of the train set
    pandas.DataFrame
        DataFrame with the paths to the input images, binary images and the targets
        of the validation set
    pandas.DataFrame
        DataFrame with the paths to the input images, binary images and the targets
        of the test set
    '''
    train_ratio, val_ratio, test_ratio = split

    # Special case: test_ratio = 0.0
    if test_ratio == 0.0:
        df_train, df_val = train_test_split(
            df,
            test_size=val_ratio,
            random_state=random_seed
            )
        df_test = pd.DataFrame(columns=df.columns)
        return df_train, df_val, df_test
    
    # Special case: test_ratio = 1.0
    if test_ratio == 1.0:
        df_train = pd.DataFrame(columns=df.columns)
        df_val = pd.DataFrame(columns=df.columns)
        df_test = df
        return df_train, df_val, df_test

    # All other cases: 
    # First, split the dataframe into train_val and test set
    df_train_val, df_test = train_test_split(
        df,
        test_size=test_ratio,
        random_state=random_seed
        )

    # Then, split the train set into train and validation set
    df_train, df_val = train_test_split(
        df_train_val,
        test_size=val_ratio / (train_ratio + val_ratio),
        random_state=random_seed
        )
    return df_train, df_val, df_test

# PyTorch Dataset class of the input images and targets based on the dataframe
class CentroidDataset(Dataset):
    def __init__(self, df, normalize_targets=True, transform=None):
        '''
        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame with the paths to the input images and the targets
        transform : torchvision.transforms, optional
            Transform to apply to the input images and targets, by default None
        '''
        self.df = df
        self.normalize_targets = normalize_targets
        self.transform = transform
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        eps = 1e-20
        input_filename = self.df.iloc[idx]['input']
        input = tifffile.imread(input_filename).astype(float)
        input = torch.Tensor(input)
        input = (input - input.min()) / (input.max() - input.min() + eps)
        input = unsqueeze_to_ndim(input, 4)

        if 'target' not in self.df.columns:
            sample = {
                'input': input,
                'input_filename': input_filename
                }
        else:
            target_filename = self.df.iloc[idx]['target']
            target = tifffile.imread(target_filename).astype(float)
            target = torch.Tensor(target)
            if self.normalize_targets:
                target = (target - target.min()) / (target.max() - target.min() + eps)
            target = unsqueeze_to_ndim(target, 4)

            sample = {
                'input': input,
                'target': target,
                'input_filename': input_filename,
                'target_filename': target_filename
                }

        if self.transform:
            sample = self.transform(sample)

        return sample
    

def create_model(model_type, in_channels, out_channels, final_sigmoid, f_maps, depth):
    assert model_type in ['UNet3D', 'ResidualUNet3D'], 'Model type should be either UNet3D or ResidualUNet3D'

    # Set model type
    if model_type == 'UNet3D':
        model_class = pytorch3dunet.UNet3D
    elif model_type == 'ResidualUNet3D':
        model_class = pytorch3dunet.ResidualUNet3D
    
    # Set is_segmentation if final sigmoid is needed, even though we are not doing segmentation.
    # This is needed because UNet implementation only adds a final sigmoid activation 
    # is_segmentation is True
    is_segmentation = True if final_sigmoid else False
    
    # Create the model
    model = model_class(
        in_channels=in_channels,
        out_channels=out_channels,
        is_segmentation=is_segmentation, 
        final_sigmoid=final_sigmoid,
        f_maps=f_maps,
        num_levels=depth
    )
    return model


def MS_SSIM(output, target):
    loss = 1 - ms_ssim(output, target, data_range=1.0, size_average=True)
    return loss


def test_model(model, dataloader, device, criterion, tqdm_desc='Testing'):
    '''
    Test the model on the test set

    Parameters
    ----------
    model : torch.nn.Module
        Model to test
    dataloader : torch.utils.data.DataLoader
        Dataloader of the test set
    device : torch.device
        Device to use for the model
    criterion : torch.nn.Module
        Loss function to use for the test set

    Returns
    -------
    float
        Loss of the test set
    
    '''
    model.eval()
    test_loss = 0
    n=0
    with torch.no_grad():
        with tqdm(dataloader, desc=tqdm_desc, leave=False, unit='batch') as dataloader:
            for batch in dataloader:
                input = batch['input'].to(device)
                target = batch['target'].to(device)
                n += 1 # Count number of batches, to compute average loss on whole test set

                # Check if input and target Tensors do not contain NaNs or Infs
                if not torch.isfinite(input).all():
                    raise ValueError(f"One of the following inputs contains NaN or Inf: {batch['input_filename']}")
                if not torch.isfinite(target).all():
                    raise ValueError(f"One of the following targets contains NaN or Inf: {batch['target_filename']}")

                with torch.cuda.amp.autocast():
                    output = model(input)
                    if not torch.isfinite(output).all():
                        raise ValueError(f"Output contains NaN or Inf \n input: {input} \n \n output: {output}")
                    loss = criterion(output, target)
                    if not torch.isfinite(loss).all():
                        raise ValueError(f"Loss contains NaN or Inf: {loss}")
                test_loss += loss.item()
                dataloader.set_postfix(test_loss=loss.item())
    return test_loss/n


def train_model(model, train_dataloader, val_dataloader, device, \
                criterion, optimizer, scheduler, config):
    '''
    Train the model

    Parameters
    ----------
    model : torch.nn.Module
        Model to train
    train_dataloader : torch.utils.data.DataLoader
        Dataloader of the train set
    val_dataloader : torch.utils.data.DataLoader
        Dataloader of the validation set
    device : torch.device
        Device to use for the model
    criterion : torch.nn.Module
        Loss function to use for the train and validation set
    optimizer : torch.optim.Optimizer
        Optimizer to use for the model
    scheduler : torch.optim.lr_scheduler
        Learning rate scheduler to use for the model
    config : dict
        Dictionary containing the configurations for training the model
    '''
    writer = SummaryWriter(log_dir=Path(config['output_path']).joinpath('runs', config['time_stamp']))

    model.to(device)
    train_loss = []
    val_loss = []
    best_val_loss = np.inf

    scaler = torch.cuda.amp.GradScaler()
    num_epochs = config['num_epochs']
    for epoch in range(num_epochs):
        epoch_train_loss = 0
        epoch_val_loss = 0
        n=0

        # Train
        model.train()
        with tqdm(train_dataloader, desc=f'Epoch {epoch +1}/{num_epochs} - Training', leave=False, unit='batch') as dataloader:
            for batch in dataloader:
                input = batch['input'].to(device)
                target = batch['target'].to(device)
                n += 1 # Count number of batches, to compute average loss per epoch

                # Check if input and target Tensors do not contain NaNs or Infs
                if not torch.isfinite(input).all():
                    raise ValueError(f"One of the following inputs contains NaN or Inf: {batch['input_filename']}")
                if not torch.isfinite(target).all():
                    raise ValueError(f"One of the following targets contains NaN or Inf: {batch['target_filename']}")
                
                optimizer.zero_grad()
                with torch.cuda.amp.autocast():
                    output = model(input)
                    loss = criterion(output, target)
                scaler.scale(loss).backward()
                
                if config['max_grad_norm'] is not None:
                    # Unscale the gradients of optimizer's assigned params in-place
                    scaler.unscale_(optimizer)
                    # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])
                
                # optimizer's gradients are already unscaled, so scaler.step does not unscale them,
                # although it still skips optimizer.step() if the gradients contain infs or NaNs.
                scaler.step(optimizer)

                # Updates the scale for next iteration.
                scaler.update()
                epoch_train_loss += loss.item()
                dataloader.set_postfix(loss=loss.item())
        epoch_train_loss = epoch_train_loss/n
        train_loss.append(epoch_train_loss)
        
        # Validation
        epoch_val_loss = test_model(
            model, val_dataloader, device, criterion, tqdm_desc= f'Epoch {epoch +1}/{num_epochs} - Validating'
            )
        scheduler.step(epoch_val_loss)
        val_loss.append(epoch_val_loss)
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_val_epoch = epoch
            best_model_state = model.state_dict()
            filename = Path(config['output_path']).joinpath(f"{config['time_stamp']}_best_model.pt")
            torch.save(best_model_state, filename)
        writer.add_scalars(
            'Loss',
            {'Train': epoch_train_loss,'Val': epoch_val_loss},
            epoch
        )
        writer.flush()
        print(f'Epoch {epoch +1}/{num_epochs} - Train loss: {epoch_train_loss:.7f} - Val loss: {epoch_val_loss:.7f}')
        
        # Save current model state
        filename = Path(config['output_path']).joinpath(f"{config['time_stamp']}_last_model.pt")
        torch.save(model.state_dict(), filename)

        # Save current config
        config['train_loss'] = train_loss
        config['val_loss'] = val_loss
        config['best_val_epoch'] = best_val_epoch
        config['best_val_loss'] = val_loss[best_val_epoch]
        config['epoch'] = epoch
        with open(config['filename'] , 'w') as f:
            json.dump(config, f, indent=4)
        
    writer.close()
    return best_model_state, train_loss, val_loss, best_val_epoch

def main(**kwargs):
    # Create time stamp for the current run
    time_stamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Create the output directory
    output_path = Path(kwargs['output_path'])
    output_path.mkdir(exist_ok=True)

    # Setup logging
    logfile = output_path.joinpath(f'{time_stamp}.log')
    file_handler = logging.FileHandler(filename=logfile, mode='w')
    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    handlers = [file_handler, stdout_handler]

    logging.basicConfig(
        level=logging.DEBUG, 
        format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s',
        handlers=handlers
    )

    try:

        # Store the config in a json file
        config = {key: (value if not isinstance(value, Path) else str(value)) for key, value in kwargs.items()} 
        config['time_stamp'] = time_stamp
        config['filename'] = str(output_path.joinpath(f'{time_stamp}_config.json'))
        with open(config['filename'] , 'w') as f:
            json.dump(config, f, indent=4)

        # Create a dataframe with the paths to the input images, binary images and the targets
        df = get_data(kwargs['input_path'], kwargs['binary_path'], kwargs['target_path'], check_batches=kwargs['check_batches'])

        # Filter out the patches that do not have enough foreground voxels
        if kwargs['binary_path'] is not None and kwargs['min_percentage'] is not None:
            assert isinstance(kwargs['min_percentage'], (float, int))
            assert kwargs['min_percentage'] > 0 and kwargs['min_percentage'] <= 1, 'The min_percentage should be between 0 and 1'
            df = get_foreground_percentage(df)
            df = df[df['foreground_percentage']>=kwargs['min_percentage']]
        print('Number of patches:', len(df))
        
        # Split dataframe into train, validation and test in which test set is only consisting of one batch
        train_ratio, val_ratio, test_ratio = kwargs['split']
        df_train, df_val, df_test = split_dataframe(
            df,
            kwargs['split'],
            random_seed = kwargs['random_seed']
        )
        df_train['split'] = 'train'
        df_val['split'] = 'val'
        df_test['split'] = 'test'

        print('Number of patches in train set:', len(df_train))
        print('Number of patches in validation set:', len(df_val))
        print('Number of patches in test set:', len(df_test))
        print()

        # Concatenate train, validation and test dataframes and store them in a csv file    
        df = pd.concat([df_train, df_val, df_test])
        df.to_csv(output_path.joinpath(f'{time_stamp}_split.csv'), index=False)

        # Create a transform to apply to the input images and targets
        transforms = Compose([
            RandomFlip3D(axis=-1, p=0.5),
            RandomFlip3D(axis=-2, p=0.5),
            RandomFlip3D(axis=-3, p=0.5),
            AugmentGaussianNoise(
                sigma=kwargs['augment_noise_sigma'],
                p=kwargs['augment_noise_p']
            ),
            AugmentBrightness(
                mu=0.0,
                sigma=kwargs['augment_brightness_sigma'],
                p_per_channel=0.5
                ),
            AugmentContrast(
                contrast_range=(
                    kwargs['augment_contrast_range'][0],
                    kwargs['augment_contrast_range'][1]
                    ),
                p_per_channel=0.5
                ),
            RandomRescaledCrop3D(
                scale_range=(
                    kwargs['augment_rescale_range'][0],
                    kwargs['augment_rescale_range'][1]
                    ),
                anisotropic=kwargs['augment_rescale_anisotropic'],
                p=kwargs['augment_rescale_p']
            ),
        ])

        # Create a dictionary with train, validation and test datasets
        datasets = {
            'train': CentroidDataset(df_train, normalize_targets=kwargs['normalize_targets'], transform=transforms),
            'val': CentroidDataset(df_val, normalize_targets=kwargs['normalize_targets']),
            'test': CentroidDataset(df_test, normalize_targets=kwargs['normalize_targets'])
        }

        # Create a dictionary with train, validation and test dataloaders
        generator = set_random_seeds(seed=kwargs['random_seed'])
        batch_size = kwargs['batch_size']
        dataloaders = {
            'train': DataLoader(
                datasets['train'],
                batch_size=batch_size,
                shuffle=True if train_ratio > 0.0 else False, # Gives an error otherwise of test_ratio == 1.0
                num_workers=0,
                generator=generator,
                drop_last=False,
                worker_init_fn=seed_worker
                ),
            'val': DataLoader(
                datasets['val'],
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                generator=generator,
                drop_last=False,
                worker_init_fn=seed_worker
                ),
            'test': DataLoader(
                datasets['test'],
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                generator=generator,
                drop_last=False,
                worker_init_fn=seed_worker
                )
        }

        # Set device
        device = torch.device(f"cuda:{kwargs['gpu_id']}" if torch.cuda.is_available() else "cpu")

        # Create the model
        model = create_model(
            model_type=kwargs['model_type'],
            in_channels=kwargs['in_channels'],
            out_channels=kwargs['out_channels'],
            final_sigmoid=kwargs['final_sigmoid'],
            f_maps=kwargs['f_maps'],
            depth=kwargs['depth']
        )
        model.to(device)

        # Load pretrained model or train new one
        if kwargs['pretrained'] is not None: 
            assert Path(kwargs['pretrained']).is_file() and \
                Path(kwargs['pretrained']).suffix.endswith('.pt'), \
                'The pretrained model should be an existing .pt file'
            # Load pretrained model if available
            print('Loading pretrained model')
            model.load_state_dict(torch.load(kwargs['pretrained']))

        # Train the model
        if kwargs['loss'] == 'MSE':
            criterion = nn.MSELoss(reduction='mean')
        elif kwargs['loss'] == 'MS_SSIM':
            criterion = MS_SSIM
        elif kwargs['loss'] == 'MSE_MS_SSIM':
            criterion = lambda output, target: 100*nn.MSELoss(reduction='mean')(output, target) + MS_SSIM(output, target)
        elif kwargs['loss'] == 'TverskyLoss':
            criterion = TverskyLoss(binarize_targets=True)
        else:
            raise ValueError('The loss function is not recognized')
        if test_ratio < 1.0 and not kwargs['test_only']:
            print('Training the model')
            optimizer = torch.optim.Adam(model.parameters(), lr=kwargs['lr'])
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=kwargs['patience'], verbose=True
                )
            torch.autograd.set_detect_anomaly(kwargs['detect_anomaly'])
            best_model_state, _, _, _ = train_model(
                model,
                dataloaders['train'],
                dataloaders['val'],
                device,
                criterion,
                optimizer,
                scheduler,
                config
                )
            model.load_state_dict(best_model_state)
        
        # Test the model
        if test_ratio > 0.0:
            print('Testing the model')
            test_loss = test_model(model, dataloaders['test'], device, criterion, tqdm_desc='Testing')
            print(f'Test loss: {test_loss:.7f}')

            # Add test results to the config and save
            config['test_loss'] = test_loss

        # Save config as json file
        with open(config['filename'] , 'w') as f:
            json.dump(config, f, indent=4)

    except Exception as e:
        logging.error('%s \n %s', e, traceback.format_exc())

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--input_path', type=str, default='../data/02a_patches', 
        help='Path to the input patches')
    parser.add_argument('--binary_path', type=str, default='../data/02b_patches_binarized',
        help='Path to the binary patches')
    parser.add_argument('--target_path', type=str, default='../data/03_weak_targets',
        help='Path to the target patches')
    parser.add_argument('--output_path', type=str, default='../data/training_output',
        help='Path to the output folder')
    parser.add_argument('--normalize_targets', action='store_true',
        help='Whether to normalize the target images or not (default: False)')
    parser.add_argument('--split', nargs=3, type=float, default=[0.7, 0.1, 0.2],
        help='Train, validation and test split (default: 0.7, 0.1, 0.2)')
    parser.add_argument('--random_seed', type=int, default=0,
        help='Random seed (default: 0)')
    parser.add_argument('--min_percentage', type=float, default=None,
        help='Filter for minimum percentage of foreground in the input images (default: None). If None, no filtering is performed')
    parser.add_argument('--batch_size', type=int, default=2,
        help='Batch size (default: 2)')
    parser.add_argument('--num_epochs', type=int, default=100,
        help='Number of epochs (default: 100)')
    parser.add_argument('--lr', type=float, default=1e-4,
        help='Learning rate (default: 1e-4)')
    parser.add_argument('--max_grad_norm', type=float, default=None,
        help='Maximum gradient norm (default: None, no clipping) for gradient clipping')
    parser.add_argument('--model_type', type=str, choices=('UNet3D', 'ResidualUNet3D') , default='UNet3D',
        help='Model type (default: UNet3D)')
    parser.add_argument('--pretrained', type=str, default=None,
        help=('Define path of saved pretrained model state dict. '
              'By default, no pretrained model is used and a new model is trained'))
    parser.add_argument('--final_sigmoid', action='store_true', default=False,
        help='Whether to use a sigmoid activation function in the final layer (default: True)')
    parser.add_argument('--loss', type=str, choices=['MSE', 'MSE_MS_SSIM','MSE_MS_SSIM', 'TverskyLoss'], default='MSE',
        help='Loss function (default: MSE)')
    parser.add_argument('--patience', type=int, default=5,
        help='Patience for ReduceLROnPlateau scheduler (default: 5)')
    parser.add_argument('--augment_rescale_p', type=float, default=0.5,
        help='Probability to using rescale data augmentation (default: False)')
    parser.add_argument('--augment_rescale_range', nargs=2, type=float, default=[0.75, 1.25],
        help='Rescale range for rescaling augmentation (default: 0.75, 1.25)')
    parser.add_argument('--augment_rescale_anisotropic', action='store_true', default=False,
        help='Whether to rescale anisotropically or not during data augmentation (default: False)')
    parser.add_argument('--augment_brightness_sigma', type=float, default=0.1,
        help='Sigma for brightness augmentation (default: 0.1)')
    parser.add_argument('--augment_contrast_range', nargs=2, type=float, default=[0.9, 1.1],
        help='Contrast range for contrast augmentation (default: 0.9, 1.1)')
    parser.add_argument('--augment_noise_sigma', type=float, default=0.1,
        help='Sigma for Gaussian noise augmentation (default: 0.1)')
    parser.add_argument('--augment_noise_p', type=float, default=0.5,
        help='Probability for Gaussian noise augmentation (default: 0.5)')
    parser.add_argument('--in_channels', type=int, default=1,
        help='Number of input channels (default: 1)')
    parser.add_argument('--out_channels', type=int, default=1,
        help='Number of output channels (default: 1)')
    parser.add_argument('--f_maps', type=int, default=16,
        help='Number of feature maps (default: 16)')
    parser.add_argument('--depth', type=int, default=4,
        help='Depth of the UNet (default: 4)')
    parser.add_argument('--gpu_id', type=int, default=0, 
        help='GPU id to use (default: 0)')
    parser.add_argument('--detect_anomaly', action='store_true', default=False,
        help='Whether to detect anomaly or not (default: False)')
    parser.add_argument('--test_only', action='store_true', default=False,
        help='Whether to only test the model or not (default: False)')
    parser.add_argument('--check_batches', action='store_true',
        help='Whether to check for batches (parent directories for image files) or not (default: False)')
    args, _ = parser.parse_known_args()

    assert sum(args.split) == 1.0, 'The sum of the split values should be 1'
    assert args.batch_size > 0, 'The batch size should be greater than 0'
    assert args.lr > 0, 'The learning rate should be greater than 0'
    assert args.num_epochs > 0, 'The number of epochs should be greater than 0'
    assert args.augment_brightness_sigma >= 0, 'The brightness sigma should be greater than or equal to 0'
    assert args.augment_contrast_range[0] > 0 and args.augment_contrast_range[1] > 0, 'The contrast range should be greater than 0'

    return args

if __name__ == '__main__':
    args = parse_args()
    main(**vars(args))