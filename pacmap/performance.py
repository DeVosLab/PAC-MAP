import os
from dotenv import load_dotenv
from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm
import tifffile
import numpy as np
import pandas as pd
from skimage.measure import regionprops_table, label
from skimage.segmentation import clear_border
from scipy.spatial import distance
from scipy.optimize import linear_sum_assignment

from utils import merge_close_points
from prob2points import get_points
from points2df import get_points_by_method


def intensity_sum(regionmask, region_intensity):
    return np.where(regionmask, region_intensity, 0).sum()


def get_metrics(gt_points, pred_points, max_dist):
    # Get number of points
    n_gt = len(gt_points)
    n_pred = len(pred_points)

    # Calculate distances
    if n_gt == 0 or n_pred == 0:
        distances = np.zeros((0,0))
    else:
        distances = distance.cdist(
            gt_points,
            pred_points,
            'sqeuclidean'
            )
    
    # Fill with dummy distance if too large
    max_dist = max_dist**2
    dummy_dist = np.max((distances.max(), max_dist + 1)) # Dummy distance larger than max_dist_um
    if distances.size > 0:
        distances[distances > max_dist] = dummy_dist

    # Match points using linear sum assignment
    row_inds, col_inds = linear_sum_assignment(distances)
    good_match = distances[row_inds, col_inds] <= max_dist
    row_inds = row_inds[good_match]
    col_inds = col_inds[good_match]

    # Calculate metrics
    tp = len(row_inds)
    fp = n_pred - tp
    fn = n_gt - tp
    eps = 1e-20
    acc = tp/(tp+fp+fn + eps)
    precision = tp/(tp+fp + eps)
    recall = tp/(tp+fn + eps)
    f1 = 2*precision*recall/(precision+recall + eps)
    return tp, fp, fn, acc, precision, recall, f1


def get_metrics_boutin(pred_points, gt_points, max_dist=6):
    """
    Compute true positives, false positives and false negatives based ground truth 
    and predicted point coordinates as described in Boutin et al. (2018).

    Parameters
    ----------
    pred_points : np.ndarray
        Array of shape (n, 3) containing the coords of the predicted points.
    gt_points : np.ndarray
        Array of shape (m, 3) containing the coords of the ground truth points.
    max_dist : int, optional
        Radius of the neighborhood in which a centroid is considered a true
        positive, by default 6
    
    Returns
    -------
    tp : int
        Number of true positives.
    fp : int
        Number of false positives.
    fn : int
        Number of false negatives.
    """

    tp = 0
    fp = 0
    fn = 0

    nsb = len(pred_points)  # Number of centroids determined by segmentation
    ngt = len(gt_points)   # Number of centroids in the GT

    for gt_point in gt_points:
        # Find centroids in the segmentation within the neighborhood of the GT centroid
        distances = distance.cdist(
            np.array([gt_point]),
            pred_points
            )
        within_radius = np.where(distances <= max_dist)[1]

        if len(within_radius) == 1:
            # Exactly one centroid found within the neighborhood, count as TP
            closest_index = np.argmin(distances)
            tp += 1
            # Remove the matched centroid from the list
            pred_points = np.delete(pred_points, closest_index, axis=0)
        elif len(within_radius) > 1:
            # More than one centroid found within the neighborhood, choose the closest one as TP
            closest_index = np.argmin(distances)
            tp += 1
            # Remove the matched centroid from the list
            pred_points = np.delete(pred_points, closest_index, axis=0)

    # Calculate FP and FN based on TP
    fp = nsb - tp
    fn = ngt - tp

    return tp, fp, fn


def main(**kwargs):
    # Check arguments
    assert kwargs['true_path'] is not None, 'Please provide the path to the ground truths'
    
    assert kwargs['points_path'] is not None or kwargs['preds_path'] is not None,\
        'Please provide the paths to the points or raw predictions'
    
    if kwargs['intensity_as_spacing']:
        assert kwargs['preds_path'] is not None, 'Please provide the path to the raw predictions'

    # Load environment variables
    load_dotenv()
    data_path = Path(os.getenv('DATA_PATH'))

    voxelsize = kwargs['voxelsize']
    bordersize_um = kwargs['bordersize_um']
    volumesize_vox = kwargs['volumesize']

    input_path_points = data_path.joinpath(kwargs['points_path']) if kwargs['points_path'] is not None else None
    input_path_preds = data_path.joinpath(kwargs['preds_path']) if kwargs['preds_path'] is not None else None
    input_path_true = data_path.joinpath(kwargs['true_path']) if kwargs['true_path'] is not None else None
    input_path_imgs = data_path.joinpath(kwargs['img_path']) if kwargs['img_path'] is not None else None
    input_path_binary = data_path.joinpath(kwargs['binary_path']) if kwargs['binary_path'] is not None else None

    assert input_path_points is not None or input_path_preds is not None, \
        'Please provide the paths to the points or raw predictions'

    if kwargs['min_foreground'] > 0:
        assert input_path_binary is not None, 'Please provide the path to the binary masks'

    # Get all masks files
    if kwargs['true_path'] is not None:
        if kwargs['true2points_method'] == 'masks':
            suffix = '.tif'
        elif kwargs['true2points_method'] == 'npy':
            suffix = '.npy'
        elif kwargs['true2points_method'] == 'csv':
            suffix = '.csv'
        files_true = sorted([f for f in input_path_true.rglob('*') if f.is_file() and f.suffix == suffix and not f.stem.startswith('.')])
    else:
        raise ValueError('Please provide the path to the masks')

    # Get all points files
    if kwargs['points_path'] is not None:
        files_points = sorted([f for f in input_path_points.rglob('*') if f.is_file() and f.suffix == '.csv' and not f.stem.startswith('.')])
        files_points = [f for f in files_points if f'{f.parent.stem}/{f.stem}' in [f'{file_true.parent.stem}/{file_true.stem}' for file_true in files_true]]
    else:
        files_points = [None] * len(files_true)
    
    # Get all preds files
    if kwargs['preds_path'] is not None:
        files_preds = sorted([f for f in input_path_preds.rglob('*') if f.is_file() and f.suffix == '.tif' and not f.stem.startswith('.')])
        files_preds = [f for f in files_preds if f'{f.parent.stem}/{f.stem}' in [f'{file_true.parent.stem}/{file_true.stem}' for file_true in files_true]]
    else:
        files_preds = [None] * len(files_true)
    
    # Get all img files
    if kwargs['img_path'] is not None:
        files_imgs = sorted([f for f in input_path_imgs.rglob('*') if f.is_file() and f.suffix == '.tif' and not f.stem.startswith('.')])
        files_imgs = [f for f in files_imgs if f'{f.parent.stem}/{f.stem}' in [f'{file_true.parent.stem}/{file_true.stem}' for file_true in files_true]]
    else:
        files_imgs = [None] * len(files_true)

    # Get all binary files
    if kwargs['binary_path'] is not None:
        files_binary = sorted([f for f in input_path_binary.rglob('*') if f.is_file() and f.suffix == '.tif' and not f.stem.startswith('.')])
        files_binary = [f for f in files_binary if f'{f.parent.stem}/{f.stem}' in [f'{file_true.parent.stem}/{file_true.stem}' for file_true in files_true]]
    else:
        files_binary = [None] * len(files_true)

    # Loop over all files
    df = pd.DataFrame()
    for file_points, file_preds, file_true, file_img, file_binary in tqdm(zip(files_points, files_preds, files_true, files_imgs, files_binary), total=len(files_true)):
        # Get foreground fraction, if provided
        if file_binary is not None:
            binary = tifffile.imread(file_binary)
            foreground_fraction = binary.sum() / binary.size
        else:
            foreground_fraction = None
        
        # Skip if foreground fraction is below threshold
        if foreground_fraction is not None and foreground_fraction < kwargs['min_foreground']:
            continue
        
        # Border edge mask
        bordersize_vox = np.ceil(np.array([bordersize_um / v for v in voxelsize])).astype(int)
        border_mask = np.ones(volumesize_vox, dtype=bool)
        border_mask[
            bordersize_vox[0]:-bordersize_vox[0],
            bordersize_vox[1]:-bordersize_vox[1],
            bordersize_vox[2]:-bordersize_vox[2]
            ] = False
        
        # Clear border masks
        if kwargs['true2points_method'] == 'masks':
            true_mask = tifffile.imread(file_true)
            foreground = true_mask > 0
            true_mask = clear_border(true_mask, mask=~border_mask)
            border_touching_mask = np.logical_xor(true_mask, foreground)
            border_mask = np.logical_or(border_mask, border_touching_mask)
        else:
            true_points, _ = get_points_by_method(file_true, kwargs['true2points_method'])
            true_points = true_points.astype(int)
            true_points_img = np.zeros(volumesize_vox)
            for p in true_points:
                true_points_img[p[0], p[1], p[2]] = 1
            true_points_img = np.where(border_mask, 0, true_points_img)
            true_points = np.argwhere(true_points_img)

        # Get points
        if file_points is not None:
            # Get points from csv file
            points = pd.read_csv(file_points, index_col=0, header=0).values.astype(int)
        elif file_preds is not None:
            # Get points from raw predictions
            preds = tifffile.imread(file_preds)
            if kwargs['pred2points_method'] == 'peaks':
                points = get_points(
                    preds,
                    min_distance=kwargs['min_distance'],
                    threshold_abs=kwargs['threshold'],
                    exclude_border=False,
                    intensity_as_spacing=kwargs['intensity_as_spacing'],
                    top_down=True if not kwargs['intensity_as_spacing'] or kwargs['top_down'] else False,
                    voxelsize=voxelsize
                )
            elif kwargs['pred2points_method'] == 'cc':
                preds_bw = preds > kwargs['threshold']
                preds_cc = label(preds_bw)
                props_preds = regionprops_table(preds_cc, properties=['label', 'centroid'])
                df_props_preds = pd.DataFrame(props_preds)
                points = df_props_preds[['centroid-0', 'centroid-1', 'centroid-2']].values.astype(int)
            elif kwargs['pred2points_method'] in ['npy', 'csv', 'masks']:
                points, _ = get_points_by_method(file_preds, kwargs['pred2points_method']).astype(int)
            else:
                raise ValueError('Invalid method to convert predictions to points')

        # Clear border points
        pred_points_img = np.where(border_mask, 0, pred_points_img)
        points = np.argwhere(pred_points_img)

        # Merge close points
        if kwargs['merge']:
            points = merge_close_points(
                points,
                voxelsize=voxelsize,
                threshold=kwargs['min_distance'],
                ).astype(int)
        n_pred_points = len(points)

        # Skip if no points
        if len(true_points) == 0 or len(points) == 0:
            continue

        # Create points image
        pred_points_img = np.zeros(kwargs['volumesize'], dtype=bool)
        for p in points:
            pred_points_img[p[0], p[1], p[2]] = 1

        # Calculate precision and recall
        if kwargs['score_method'] == 'own':
            # Own method

            # Convert points to um
            points_um = points * np.array(voxelsize)
            true_points_um = true_points * np.array(voxelsize)
            n_true_points = len(true_points)

            # Calculate metrics
            tp, fp, fn, _, precision, recall, _, _ = get_metrics(
                true_points_um,
                points_um,
                max_dist=kwargs['max_dist_um']
            )
        elif kwargs['score_method'] == 'krupa' and kwargs['true2points_method'] == 'masks':
            # Krupa et al. (2021) method

            # Get the number of pred points in each true mask
            props = regionprops_table(
                true_mask,
                intensity_image=pred_points_img,
                properties=['label', 'centroid'],
                extra_properties=(intensity_sum,)
            )
            df_props = pd.DataFrame(props)
            n_true_points = len(df_props)
            df_props['correct'] = df_props['intensity_sum'] >= 1 # The masks with at least 1 points
            df_props['missed'] = df_props['intensity_sum'] == 0
            
            # Calculate metrics
            tp = df_props['correct'].sum()
            fp = n_pred_points - tp
            fn = df_props['missed'].sum()
            precision = tp / n_pred_points
            recall = tp / n_true_points

        elif kwargs['score_method'] == 'boutin':
            # Boutin et al. (2018) method

            # Convert points to um
            points_um = points * np.array(voxelsize)
            true_points_um = true_points * np.array(voxelsize)
            n_true_points = len(true_points)

            # Calculate metrics
            tp, fp, fn = get_metrics_boutin(
                pred_points=points_um,
                gt_points=true_points_um,
                max_dist=kwargs['max_dist_um'],
            )
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
        else:
            raise ValueError('Invalid score method')
        
        # Save results to df
        df_entry = pd.DataFrame({
            'file':                 [file_true.stem],
            'foreground_fraction':  [foreground_fraction],
            'n_true_points':        [n_true_points],
            'n_pred_points':        [n_pred_points],
            'TPR':                  [tp/n_pred_points],
            'FPR':                  [fp/n_pred_points],
            'FNR':                  [fn/n_true_points],
            'precision':            [precision],
            'recall':               [recall],
            'f1':                   [2 * precision * recall / (precision + recall + 1e-6)],
        })
        df = pd.concat(
            [df if not df.empty else None, df_entry],
            axis=0
        ).reset_index(drop=True)

    # Save results
    if kwargs['output_path'] is not None:
        output_path = data_path.joinpath(kwargs['output_path'])
        output_path.mkdir(exist_ok=True, parents=True)
        df.to_csv(output_path.joinpath(f"{kwargs['filename']}.csv"), index=False)

    # Print results
    # Get average +/- std of precision and recall
    tpr_mean = df['TPR'].mean()
    tpr_std = df['TPR'].std()
    fpr_mean = df['FPR'].mean()
    fpr_std = df['FPR'].std()
    fnr_mean = df['FNR'].mean()
    fnr_std = df['FNR'].std()
    precision_mean = df['precision'].mean()
    precision_std = df['precision'].std()
    recall_mean = df['recall'].mean()
    recall_std = df['recall'].std()
    f1_mean = df['f1'].mean()
    f1_std = df['f1'].std()
    avg_dist_mean = df['avg_dist'].mean()
    avg_dist_std = df['avg_dist'].std()
    print(f'TPR: {tpr_mean:.3f} +/- {tpr_std:.3f}')
    print(f'FPR: {fpr_mean:.3f} +/- {fpr_std:.3f}')
    print(f'FNR: {fnr_mean:.3f} +/- {fnr_std:.3f}')
    print(f'Precision: {precision_mean:.3f} +/- {precision_std:.3f}')
    print(f'Recall: {recall_mean:.3f} +/- {recall_std:.3f}')
    print(f'F1: {f1_mean:.3f} +/- {f1_std:.3f}')
    print(f'Average distance: {avg_dist_mean:.3f} +/- {avg_dist_std:.3f}')


def parse_args():
    parser = ArgumentParser(description='Performance evaluation of predicted points.')
    parser.add_argument('--points_path', type=str, default=None,
        help='Path to the points')
    parser.add_argument('--preds_path', type=str, default=None,
        help='Path to the predictions')
    parser.add_argument('--true_path', type=str, default=None,
        help='Path to the masks')
    parser.add_argument(
        '--img_path', type=str, default=None,
        help='Path to the images')
    parser.add_argument('--binary_path', type=str, default=None,
        help='Path to the binary masks, used to calculate foreground fraction')
    parser.add_argument('--output_path', type=str, default=None,
        help='Path to save the results')
    parser.add_argument('--filename', type=str, default='performance',
        help='Filename of the results without extension')
    parser.add_argument('-v', '--voxelsize', type=float, nargs=3, default=[1.9999, 0.3594, 0.3594],
        help='Voxelsize of the images')
    parser.add_argument('-b', '--bordersize_um', type=float, default=5,
        help='Bordersize in um')
    parser.add_argument('--volumesize', type=int, nargs=3, default=[46, 256, 256],
        help='Volume size in voxels')
    parser.add_argument('--true2points_method', type=str, choices=['npy', 'csv', 'masks'], default='csv',
        help='Method to convert ground truth to points')
    parser.add_argument('--pred2points_method', type=str, choices=['peaks', 'cc', 'csv', 'masks'], default='peaks',
        help='Method to convert predictions to points')
    parser.add_argument('-m', '--merge', action='store_true',
        help='Merge close points')
    parser.add_argument('-d', '--min_distance', type=float, default=5,
        help='Minimum distance between points (default: 5)')
    parser.add_argument('--max_dist_um', type=float, default=5,
        help='Maximum distance in um for points to be considered a match (default: 5)')
    parser.add_argument('-t', '--threshold', type=float, default=0.1,
        help='Threshold for peak detection (default: 0.1)')
    parser.add_argument('-i', '--intensity_as_spacing', action='store_true',
        help='Use intensity as spacing')
    parser.add_argument('--top_down', action='store_true',
        help='Use top down approach for peak detection')
    parser.add_argument('-s', '--score_method', type=str, default='own', choices=['own', 'krupa', 'boutin'], 
        help='Specify the scoring method to be used (default: own)')
    parser.add_argument('--min_foreground', type=float, default=0.0,
        help='Minimum foreground fraction')
    args, _ = parser.parse_known_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    main(**vars(args))