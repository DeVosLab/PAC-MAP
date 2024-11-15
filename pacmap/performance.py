import os
from dotenv import load_dotenv
from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm
import tifffile
import numpy as np
import pandas as pd
from skimage.morphology import ball
from skimage.transform import resize
from skimage.measure import regionprops_table, label
from skimage.segmentation import clear_border
from scipy.interpolate import interp1d
from scipy.spatial import distance
from scipy.optimize import linear_sum_assignment

from .utils import merge_close_points
from .prob2points import get_points
from .points2df import get_points_by_method
from .peaks import _get_peak_mask, peak_local_max


def intensity_sum(regionmask, region_intensity):
    return np.where(regionmask, region_intensity, 0).sum()


def get_fixed_thresholds(t_min=0, t_max=1, n=100):
    return np.linspace(t_min, t_max, n)

def interpolate_pr(thresholds, precisions, recalls, fixed_thresholds=get_fixed_thresholds()):
    precision_interp = interp1d(thresholds, precisions, bounds_error=False, fill_value=(precisions[0], precisions[-1]))
    recall_interp = interp1d(thresholds, recalls, bounds_error=False, fill_value=(recalls[0], recalls[-1]))
    
    return precision_interp(fixed_thresholds), recall_interp(fixed_thresholds)

def calculate_auc(thresholds, precisions, recalls, envelope=False, method=None):
    # Interpolate precision and recall
    interp_precisions, interp_recalls = interpolate_pr(
        thresholds,
        precisions,
        recalls,
        get_fixed_thresholds(t_min=thresholds.min(), t_max=thresholds.max())
    )

    # Ensure precisions and recalls are sorted by recall
    sorted_pairs = sorted(zip(interp_recalls, interp_precisions), key=lambda x: x[0])
    recalls, precisions = zip(*sorted_pairs)
    
    # Append sentinel values
    recalls = np.concatenate(([0.], list(recalls), [1.]))
    precisions = np.concatenate(([0.], list(precisions), [0.]))
    
    # Compute the precision envelope
    if envelope:
        for i in range(len(precisions) - 1, 0, -1):
            precisions[i - 1] = max(precisions[i - 1], precisions[i])
    
    # Compute AUC
    auc = 0
    for i in range(len(recalls) - 1):
        auc += (recalls[i+1] - recalls[i]) * precisions[i+1]

    return auc

def get_metrics(gt_points, pred_points, max_dist):
    # Get number of points
    n_gt = len(gt_points)
    n_pred = len(pred_points)

    # Calculate distances
    if n_gt == 0 or n_pred == 0:
        # distances = np.zeros((0,0))
        tp, fp, fn, acc, precision, recall, f1 = None, None, None, None, None, None, None
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
        eps = 1e-8
        acc = tp/(tp+fp+fn + eps)
        precision = tp/(tp+fp + eps)
        recall = tp/(tp+fn + eps)
        f1 = 2*precision*recall/(precision+recall + eps)
    return tp, fp, fn, acc, precision, recall, f1


def evaluate_krupa(pred_points_img, true_mask, voxelsize, metrics_krupa=False):
    # Krupa et al. (2021)
    assert true_mask is not None, 'Please provide the true mask'
    
    props = regionprops_table(
        true_mask,
        intensity_image=pred_points_img,
        properties=['label', 'centroid'],
        extra_properties=(intensity_sum,)
    )
    df_props = pd.DataFrame(props)
    true_points = df_props[['centroid-0', 'centroid-1', 'centroid-2']].values.astype(int)

    df_props['correct'] = df_props['intensity_sum'] >= 1 # The masks with at least 1 points
    df_props['multiple'] = np.where(
        df_props['intensity_sum'] > 1, # The masks with multiple points
        df_props['intensity_sum'] - 1, # The number of points that are not correct within the mask
        0
    )
    df_props['missed'] = df_props['intensity_sum'] == 0 # The masks with no points

    mask_missed = np.zeros_like(true_mask)
    for i, row in df_props.iterrows():
        if row['missed']:
            mask_missed[true_mask == row['label']] = 1
    
    tp = df_props['correct'].sum()
    n_multiple = df_props['multiple'].sum()
    fn = df_props['missed'].sum()
    fp = np.where(true_mask == 0, pred_points_img, 0).sum() # n_pred_points - n_correct - n_multiple - n_missed

    # Get number of true and predicted points
    n_true_points = len(true_points)
    n_pred_points = pred_points_img.sum()

    # Calculate precision and recall
    if metrics_krupa:
        precision = 1 - (n_multiple + fp) / n_pred_points # As proposed by Krupa et al. (2021)
        recall = 1 - fn / n_true_points                       # As proposed by Krupa et al. (2021)
    else:
        precision = tp / n_pred_points
        recall = tp / n_true_points
    
    return tp, fp, fn, precision, recall, n_pred_points, n_true_points


def evaluate_boutin(points, true_points, voxelsize):
    # Boutin et al. (2018)

    def compute_tp_fp_fn(points, true_points, neighborhood_radius=6):
        """
        Compute true positives, false positives and false negatives based ground truth 
        and predicted point coordinates as described in Boutin et al. (2018).

        Parameters
        ----------
        points : np.ndarray
            Array of shape (n, 3) containing the coords of the predicted points.
        true_points : np.ndarray
            Array of shape (m, 3) containing the coords of the ground truth points.
        neighborhood_radius : int, optional
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

        nsb = len(points)  # Number of centroids determined by segmentation
        ngt = len(true_points)   # Number of centroids in the GT

        for true_point in true_points:
            # Find centroids in the segmentation within the neighborhood of the GT centroid
            distances = distance.cdist(np.array([true_point]), points)
            within_radius = np.where(distances <= neighborhood_radius)[1]

            if len(within_radius) == 1:
                # Exactly one centroid found within the neighborhood, count as TP
                closest_index = np.argmin(distances)
                tp += 1
                # Remove the matched centroid from the list
                points = np.delete(points, closest_index, axis=0)
            elif len(within_radius) > 1:
                # More than one centroid found within the neighborhood, choose the closest one as TP
                closest_index = np.argmin(distances)
                tp += 1
                # Remove the matched centroid from the list
                points = np.delete(points, closest_index, axis=0)

        # Calculate FP and FN based on TP
        fp = nsb - tp
        fn = ngt - tp

        return tp, fp, fn

    assert true_points is not None, 'Please provide the true points'

    # Get number of true and predicted points
    n_true_points = len(true_points)
    n_pred_points = len(points)

    # Convert points to um
    points_um = points * np.array(voxelsize)
    true_points_um = true_points * np.array(voxelsize)

    tp, fp, fn = compute_tp_fp_fn(
        pred_points=points_um,
        gt_points=true_points_um,
        neighborhood_radius=5,
    )
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    return tp, fp, fn, precision, recall, n_pred_points, n_true_points

def evaluate_own(points, true_points, voxelsize):
    # Own method
    assert true_points is not None, 'Please provide the true points'
    
    # Get number of true and predicted points
    n_true_points = len(true_points)
    n_pred_points = len(points)

    # Convert points to um
    points_um = points * np.array(voxelsize)
    true_points_um = true_points * np.array(voxelsize)
    tp, fp, fn, _, precision, recall, _ = get_metrics(true_points_um, points_um, max_dist=5)

    return tp, fp, fn, precision, recall, n_pred_points, n_true_points

def evaluate_points(
    points, border_mask, min_distance, voxelsize, 
    true_points = None, true_mask=None, 
    merge=False, score_method='own', metrics_krupa=False
    ):
    # Save original points for visualization
    points_orig = points.copy()

    # Create points image
    pred_points_img = np.zeros_like(border_mask, dtype=bool)
    for p in points:
        pred_points_img[p[0], p[1], p[2]] = 1

    # Clear border points
    pred_points_img = np.where(border_mask, 0, pred_points_img)
    points = np.argwhere(pred_points_img)

    # Merge close points
    if merge:
        points = merge_close_points(
            points,
            voxelsize=voxelsize,
            threshold=min_distance,
            ).astype(int)

    # Calculate precision and recall
    if score_method == 'krupa':
        tp, fp, fn, precision, recall, n_pred_points, n_true_points = evaluate_krupa(
            pred_points_img, true_mask, voxelsize, metrics_krupa=metrics_krupa
        )
    elif score_method == 'boutin':
        tp, fp, fn, precision, recall, n_pred_points, n_true_points = evaluate_boutin(
            points, true_points, voxelsize
        )
    elif score_method == 'own':
        tp, fp, fn, precision, recall, n_pred_points, n_true_points = evaluate_own(
            points, true_points, voxelsize
        )
    else:
        raise ValueError('Invalid score method')

    return tp, fp, fn, precision, recall, n_pred_points, n_true_points, points_orig



def main(**kwargs):
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

    # Get all true files
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

    # Create footprint for peak detection
    min_distance = kwargs['min_distance']
    se_size = min_distance # in um
    se = ball(se_size)
    se_size_vox = (se_size / np.array(voxelsize)).astype(int)
    se_size_vox = se_size_vox + (se_size_vox % 2 == 0) # Make sure that the size is odd
    footprint_vox = resize(se.astype(float), se_size_vox, order=1) > 0.5

    # Loop over all files
    df = pd.DataFrame()

    for file_points, file_preds, file_true, file_img, file_binary in zip(files_points, files_preds, files_true, files_imgs, files_binary):

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
            n_true_points = np.unique(true_mask).size - 1
        else:
            true_points, _ = get_points_by_method(file_true, kwargs['true2points_method'])
            true_points = true_points.astype(int)
            true_points_img = np.zeros(volumesize_vox)
            for p in true_points:
                true_points_img[p[0], p[1], p[2]] = 1
            true_points_img = np.where(border_mask, 0, true_points_img)
            true_points = np.argwhere(true_points_img)
            n_true_points = len(true_points)
        # Get points
        if file_points is not None:
            # Get points from csv file
            points = pd.read_csv(file_points, index_col=0, header=0).values.astype(int)

            # Skip if no points
            if len(points) == 0 or n_true_points == 0:
                continue

            # Evaluate points
            tp, fp, fn, precision, recall, n_pred_points, n_true_points, points_orig = evaluate_points(
                points, border_mask, min_distance, voxelsize,
                true_points=true_points if kwargs['score_method'] != 'krupa' else None, 
                true_mask=true_mask if kwargs['score_method'] == 'krupa'  else None, 
                merge=kwargs['merge'], score_method=kwargs['score_method'], 
                metrics_krupa=kwargs['metrics_krupa']
            )

            # Save results to df
            df_entry = pd.DataFrame({
                'method':               [kwargs['method_name']],
                'file':                 [file_true.stem],
                'foreground_fraction':  [foreground_fraction],
                'threshold':            [kwargs['threshold'][0]],
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
        elif file_preds is not None:
            # Get points from raw predictions
            preds = tifffile.imread(file_preds)
            for threshold in kwargs['threshold']:
                if kwargs['pred2points_method'] == 'peaks':
                    # Peak mask
                    peak_mask = _get_peak_mask(preds, footprint_vox, threshold)
                    points = peak_local_max(
                        preds,
                        min_distance=min_distance,
                        threshold_abs=threshold,
                        footprint=footprint_vox,
                        p_norm=2,
                        intensity_as_spacing=kwargs['intensity_as_spacing'],
                        top_down=True if not kwargs['intensity_as_spacing'] or kwargs['top_down'] else False,
                        voxelsize=voxelsize,
                        exclude_border=False
                    ).astype(int)

                elif kwargs['pred2points_method'] == 'cc' or kwargs['pred2points_method'] == 'masks':
                    if kwargs['pred2points_method'] == 'cc':
                        preds_bw = preds > threshold
                        preds_cc = label(preds_bw)
                    else:
                        preds_cc = preds.copy() # Keep preds around for visualization
                    props_preds = regionprops_table(preds_cc, properties=['label', 'centroid'])
                    df_props_preds = pd.DataFrame(props_preds)
                    points = df_props_preds[['centroid-0', 'centroid-1', 'centroid-2']].values.astype(int)
                else:
                    raise ValueError('Invalid method to convert predictions to points')

                # Skip if no points
                if len(points) == 0 or n_true_points == 0:
                    continue

                # Evaluate points
                tp, fp, fn, precision, recall, n_pred_points, n_true_points, points_orig = evaluate_points(
                    points, border_mask, min_distance, voxelsize,
                    true_points=true_points if kwargs['score_method'] != 'krupa' else None, 
                    true_mask=true_mask if kwargs['score_method'] == 'krupa'  else None, 
                    merge=kwargs['merge'], score_method=kwargs['score_method'], 
                    metrics_krupa=kwargs['metrics_krupa']
                )

                if n_pred_points == 0 or n_true_points == 0:
                    continue

                # Save results to df
                df_entry = pd.DataFrame({
                    'method':               [kwargs['method_name']],
                    'file':                 [file_true.stem],
                    'foreground_fraction':  [foreground_fraction],
                    'threshold':            [threshold],
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

    # Replace inf with nan
    df = df.replace([np.inf, -np.inf], np.nan)

    # Process results
    ## Average and std per threshold
    df_avg = df.groupby(['threshold']).mean(numeric_only=True).reset_index()
    df_std = df.groupby(['threshold']).std(numeric_only=True).reset_index()

    if len(df_avg) > 1:
        auc = calculate_auc(
            df_avg['threshold'].values,
            df_avg['precision'].values,
            df_avg['recall'].values,
            envelope=True,
            method=kwargs['method_name']
            )
    else:
        auc = None

    ## Find index of best F1 score
    idx_best_f1 = df_avg['f1'].idxmax()
    
    ## Report TPR, FPR, FNR, precision, recall, f1 of best F1 score
    tpr_mean, tpr_std = df_avg.loc[idx_best_f1, 'TPR'], df_std.loc[idx_best_f1, 'TPR']
    fpr_mean, fpr_std = df_avg.loc[idx_best_f1, 'FPR'], df_std.loc[idx_best_f1, 'FPR']
    fnr_mean, fnr_std = df_avg.loc[idx_best_f1, 'FNR'], df_std.loc[idx_best_f1, 'FNR']
    precision_mean, precision_std = df_avg.loc[idx_best_f1, 'precision'], df_std.loc[idx_best_f1, 'precision']
    recall_mean, recall_std = df_avg.loc[idx_best_f1, 'recall'], df_std.loc[idx_best_f1, 'recall']
    f1_mean, f1_std = df_avg.loc[idx_best_f1, 'f1'], df_std.loc[idx_best_f1, 'f1']
    print(f'Best threshold: {df_avg.loc[idx_best_f1, "threshold"]:.3f}')
    print(f'TPR: {tpr_mean:.3f} +/- {tpr_std:.3f}')
    print(f'FPR: {fpr_mean:.3f} +/- {fpr_std:.3f}')
    print(f'FNR: {fnr_mean:.3f} +/- {fnr_std:.3f}')
    print(f'Precision: {precision_mean:.3f} +/- {precision_std:.3f}')
    print(f'Recall: {recall_mean:.3f} +/- {recall_std:.3f}')
    print(f'F1: {f1_mean:.3f} +/- {f1_std:.3f}')
    if auc is not None:
        print(f'AUC: {auc:.3f}')
    else:
        print('AUC: N/A')

    df_summary = pd.DataFrame({
        'method':               [kwargs['method_name']],
        'TPR':                  [tpr_mean],
        'FPR':                  [fpr_mean],
        'FNR':                  [fnr_mean],
        'precision':            [precision_mean],
        'recall':               [recall_mean],
        'f1':                   [f1_mean],
        'AUC':                  [auc],
    })
    if kwargs['output_path'] is not None:
        output_path = data_path.joinpath(kwargs['output_path'])
        output_path.mkdir(exist_ok=True, parents=True)
        df_summary.to_csv(output_path.joinpath(f"{kwargs['filename']}.csv"), index=False)


def parse_args():
    parser = ArgumentParser(
        description='Performance evaluation of predicted points.'
        )
    parser.add_argument(
        '--points_path', type=str, default=None,
        help='Path to the points'
    )
    parser.add_argument(
        '--preds_path', type=str, default=None,
        help='Path to the predictions'
    )
    parser.add_argument(
        '--true_path', type=str, default=None,
        help='Path to the masks'
    )
    parser.add_argument(
        '--img_path', type=str, default=None,
        help='Path to the images'
    )
    parser.add_argument(
        '--binary_path', type=str, default=None,
        help='Path to the binary masks, used to calculate foreground fraction'
    )
    parser.add_argument(
        '--output_path', type=str, default=None,
        help='Path to save the results'
    )
    parser.add_argument(
        '--method_name', type=str, default=None,
        help='Custom name as identifier of the used method'
    )
    parser.add_argument(
        '--filename', type=str, default='performance',
        help='Filename of the results without extension'
    )
    parser.add_argument(
        '-v', '--voxelsize', type=float, nargs=3, 
        default=[1.9999, 0.3594, 0.3594],
        help='Voxelsize of the images'
    )
    parser.add_argument(
        '-b', '--bordersize_um', type=float, default=5,
        help='Bordersize in um'
    )
    parser.add_argument(
        '--volumesize', type=int, nargs=3, default=[46, 256, 256],
        help='Volume size in voxels'
    )
    parser.add_argument(
        '--true2points_method', type=str, choices=['npy', 'csv', 'masks'], 
        default='csv',
        help='Method to convert ground truth to points'
    )
    parser.add_argument(
        '--pred2points_method', type=str, choices=['peaks', 'cc', 'masks'], 
        default='peaks',
        help='Method to convert predictions to points'
    )
    parser.add_argument(
        '-m', '--merge', action='store_true',
        help='Merge close points'
    )
    parser.add_argument(
        '-d', '--min_distance', type=float, default=5,
        help='Minimum distance between points (default: 5)'
    )
    parser.add_argument(
        '-t', '--threshold', type=float, nargs='*',default=[0.1],
        help='Threshold for peak detection (default: 0.1)'
    )
    parser.add_argument(
        '-i', '--intensity_as_spacing', action='store_true',
        help='Use intensity as spacing'
    )
    parser.add_argument(
        '--top_down', action='store_true',
        help='Use top down approach for peak detection'
    )
    parser.add_argument(
        '-s', '--score_method', type=str, choices=['krupa', 'boutin', 'own'],
        default='own', 
        help='Specify the scoring method to be used (default: krupa)'
    )
    parser.add_argument(
        '--metrics_krupa', action='store_true',
        help='Calculate the metrics as proposed by Krupa et al. (2021)'
    )
    parser.add_argument(
        '--min_foreground', type=float, default=0.0,
        help='Minimum foreground fraction'
    )
    args, _ = parser.parse_known_args()

    return args

def check_args(kwargs):
    # Check arguments
    assert kwargs['true_path'] is not None, 'Please provide the path to the ground truths'
    
    assert kwargs['points_path'] is not None or kwargs['preds_path'] is not None,\
        'Please provide the paths to the points or raw predictions'
    
    if kwargs['intensity_as_spacing']:
        assert kwargs['preds_path'] is not None, 'Please provide the path to the raw predictions'

    if kwargs['min_foreground'] > 0:
        assert kwargs['binary_path'] is not None, 'Please provide the path to the binary masks'

    if isinstance(kwargs['threshold'], float):
        kwargs['threshold'] = [kwargs['threshold']]
    return kwargs

if __name__ == '__main__':
    args = parse_args()
    kwargs = vars(args)
    kwargs = check_args(kwargs)
    main(**kwargs)