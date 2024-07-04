from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm
import os
import tifffile
import numbers
import numpy as np
from skimage.transform import resize
import tensorflow as tf
from stardist.models import StarDist3D


def main(**kwargs):
    # Define input and output paths
    assert kwargs['input_path'] is not None and isinstance(kwargs['input_path'], (str, Path)), \
        'input_path must be defined'
    assert kwargs['output_path'] is not None and isinstance(kwargs['output_path'], (str, Path)), \
        'output_path must be defined'
    input_path = Path(kwargs['input_path']) # Path to folders with all the patches
    output_path = Path(kwargs['output_path']) # Path to output folder
    output_path.mkdir(parents=True, exist_ok=True)

    # Set GPU memory growth to True
    gpus = tf.config.list_physical_devices('GPU')
    if gpus and kwargs['set_memory_growth']:
        try:
            # Set memory growth on so that not all memory is allocated at once
            # Currently, however, memory growth needs to be the same across GPUs,
            # but this is not an issue if there is just a single GPU
            tf.config.experimental.set_memory_growth(gpus[kwargs['gpu_id']], True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    # Define device with GPU id
    if 'gpu_id' in range(0, len(gpus)):
        gpu_id = kwargs['gpu_id']
    else:
        gpu_id = 0
    os.environ["CUDA_VISIBLE_DEVICES"]=f"{gpu_id}"

    # Define the model
    model = StarDist3D.from_pretrained('3D_demo')

    # Optimize thresholds
    if kwargs['optimize_thresholds']:
        print('Optimizing thresholds')
        true_path = Path(kwargs['true_path'])
        files = sorted([x for x in input_path.rglob(f'*.tif') if x.is_file() and not x.stem.startswith('.')])
        files_true = sorted([x for x in true_path.rglob(f'*.tif') if x.is_file() and not x.stem.startswith('.')])

        files = [f for f in files if f.stem in [x.stem for x in files_true]]
        files_true = [f for f in files_true if f.stem in [x.stem for x in files]]

        imgs = [tifffile.imread(str(f)) for f in files]
        masks_true = [tifffile.imread(str(f)) for f in files_true]

        opt_threshs = model.optimize_thresholds(imgs, masks_true)
        print(f'Optimal thresholds: {opt_threshs}')
        kwargs['prob_thresh'] = [opt_threshs['prob']]
        kwargs['nms_thresh'] = [opt_threshs['nms']]

    # Loop over all samples
    if kwargs['check_batches']:
        samples = sorted([x for x in input_path.iterdir() if x.is_dir()])
    else:
        samples = [input_path]

    for prob_thresh in kwargs['prob_thresh']:
        for nms_thresh in kwargs['nms_thresh']:
            output_path_params = output_path.joinpath(f'prob_{prob_thresh}_nms_{nms_thresh}')
            output_path_params.mkdir(parents=True, exist_ok=True)
            for sample in tqdm(samples, colour='green'):
                # Loop over all the patches and predict
                files = sorted([x for x in sample.iterdir() if x.is_file() and x.suffix == '.tif'])
                files = files if kwargs['sample_idx'] is None else [f for i, f in enumerate(files) if i in kwargs['sample_idx']]

                print('\nPredicting')
                for file in tqdm(files, leave=False):
                    img = tifffile.imread(str(file))
                    if kwargs['img_size'] and any(img.shape/np.array(kwargs['img_size']) != 1):
                        img = resize(img, kwargs['img_size'], anti_aliasing=True)
                    img /= img.max()

                    masks, _ = model.predict_instances(
                        img,
                        prob_thresh=prob_thresh,
                        nms_thresh=nms_thresh
                        )

                    filename = output_path_params.joinpath(sample.stem, f'{file.name}')
                    filename.parent.mkdir(exist_ok=True)
                    tifffile.imwrite(filename, masks)

def parse_args():
    parser = ArgumentParser(description='Predict with StarDist')
    parser.add_argument('-i', '--input_path', type=str, default=None,
                        help='Path to input patches')
    parser.add_argument('-o', '--output_path', type=str, default=None,
                        help='Path to output folder')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='GPU id')
    parser.add_argument('--set_memory_growth', action='store_true',
                        help='Set memory growth on GPU')
    parser.add_argument('--prob_thresh', type=float, nargs='*', default=None, 
                        help='Probability threshold for stardist')
    parser.add_argument('--nms_thresh', type=float, nargs='*', default=None, 
                        help='Non-maximum suppression threshold for stardist')
    parser.add_argument('--img_size', type=int, nargs=3, default=None, 
                        help='Image size to resize to')
    parser.add_argument('--check_batches', action='store_true',
                        help='Look for images in batch subfolders')
    parser.add_argument('--sample_idx', type=int, nargs='*', default=None,
                        help='Index of files per sample to predict')
    parser.add_argument('--optimize_thresholds', action='store_true',
                        help='Optimize prob_thresh and nms_thresh thresholds')
    parser.add_argument('--true_path', type=str, default=None,
                        help='Path to true masks for optimization')
    args, _ = parser.parse_known_args()

    if isinstance(args.prob_thresh, numbers.Number) or args.prob_thresh is None:
        args.prob_thresh = [args.prob_thresh]
    if isinstance(args.nms_thresh, numbers.Number) or args.nms_thresh is None:
        args.nms_thresh = [args.nms_thresh]
    if isinstance(args.sample_idx, int):
        args.sample_idx = [args.sample_idx]
    if args.optimize_thresholds:
        assert args.true_path is not None, 'true_masks must be defined'
        args.true_path = Path(args.true_path)
        assert args.true_path.is_dir(), 'true_masks must be a directory'

    return args

if __name__ == '__main__':
    args = parse_args()
    main(**vars(args))