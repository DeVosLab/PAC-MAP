from argparse import ArgumentParser
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from performance import get_metrics


def main(**kwargs):
    # Conditional import
    if kwargs['stats']:
        from statannotations.Annotator import Annotator

    # Load dataframe
    df = pd.read_csv(kwargs['filename'], header=0, index_col=False)
    
    # Calculate differences between datasets
    df['dataset'] = df['dataset'].astype('category')
    df['input_path'] = df['input_path'].astype('category')
    df['sample'] = df['sample'].astype('category')
    df['file'] = df['file'].astype('category')
    df['axis-0'] = df['axis-0'].astype(int)
    df['axis-1'] = df['axis-1'].astype(int)
    df['axis-2'] = df['axis-2'].astype(int)

    # Define ground truth and prediction datasets
    gt_name = str(kwargs['gt_name'])
    assert gt_name in df['dataset'].unique(), f'Ground truth dataset {gt_name} not found in dataframe'
    
    pred_names = df['dataset'][df['dataset'] != gt_name].unique()
    assert len(pred_names) > 0, 'No prediction datasets found in dataframe'

    # Store results in df
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    columns = ['dataset', 'input_path', 'sample', 'file', 'n_gt', 'n_pred'] + metrics
    df_results = pd.DataFrame(columns=columns)
    voxelsize = np.array(kwargs['voxelsize'])
    for pred_name in pred_names:
        df_pred = df[df['dataset'].isin([gt_name, pred_name])]
        for group_id, group in df_pred.groupby(['sample', 'file']):
            gt_points = group[group['dataset'] == gt_name]
            pred_points = group[group['dataset'] == pred_name]

            # Skip if no points
            if len(gt_points) == 0 or len(pred_points) == 0:
                continue
            
            input_path = pred_points['input_path'].unique()[0]
            gt_points = gt_points[['axis-0', 'axis-1', 'axis-2']].values * voxelsize
            pred_points = pred_points[['axis-0', 'axis-1', 'axis-2']].values * voxelsize
            n_gt = len(gt_points)
            n_pred = len(pred_points)

            # Calculate metrics
            acc, precision, recall, f1 = get_metrics(gt_points, pred_points, kwargs['max_dist_um'])

            df_results = pd.concat([df_results, pd.DataFrame(
                [[pred_name, input_path, group_id[0], group_id[1],
                n_gt, n_pred, acc, precision, recall, f1]],
                columns=columns
                )], ignore_index=True)
    
    output_path = Path(kwargs['output_path'])
    output_path.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(output_path.joinpath('results.csv'))
    
    # Print results per dataset and store in csv
    summary = df_results.groupby(['dataset']).describe()
    summary.to_csv(output_path.joinpath('results_summary.csv'))
    for group_id, group in df_results.groupby(['dataset']):
        print(group_id)
        print(group.describe())
        print()

    # Plot results as violin plot
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(16,8))
    axes = axes.flatten() if n_metrics > 1 else [axes]
    axis_fontsize = 14
    tick_fontsize = 12
    for i, metric in enumerate(metrics):
        # Boxplot
        sns.boxplot(
            x='dataset',
            y=metric,
            data=df_results,
            ax=axes[i],
            showfliers=False,
            showmeans=True,
            meanprops={
                'marker':'o',
                'markerfacecolor':'white',
                'markeredgecolor':'black'
                },
            palette=kwargs['palette']
            )
        ylims = axes[i].get_ylim()
        axes[i].set_ylabel(f'{metric.upper().replace("_", " ")}', fontsize=axis_fontsize)
        axes[i].tick_params(axis='y', which='major', labelsize=tick_fontsize)
        axes[i].grid(True)
        
        # Strip plot
        g = sns.stripplot(
            x='dataset',
            y=metric,
            data=df_results,
            ax=axes[i],
            color='black',
            alpha=0.5,
            )

        # Set ylims
        axes[i].set_ylim(ylims)
        axes[i].set_ylim(-0.01, 1.01)
        
        # Set labels
        axes[i].set_ylabel(metric.upper().replace('_', ' '), fontsize=axis_fontsize)
        axes[i].set_xlabel('METHOD', fontsize=axis_fontsize)
        axes[i].tick_params(axis='x', rotation=90, labelsize=tick_fontsize)
        
    plt.tight_layout()
    plt.savefig(output_path.joinpath('results.png'), dpi=300)
    plt.close()

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--filename', type=str,
        help='Define filename of csv file with all points')
    parser.add_argument('--gt_name', type=str,
        help='Define the name of the ground truth dataset in the csv file')
    parser.add_argument('--output_path', type=str, default=None,
        help='Define the output path where results.csv file will be stored (optional)')
    parser.add_argument('--palette', type=str, default='Set1',
        help='Define the color palette to be used in the box plot')
    parser.add_argument('--metric', type=str, nargs='+',
        choices=['accuracy', 'precision', 'recall', 'f1'], default=['accuracy', 'precision', 'recall', 'f1'],
        help='Choose the error metric to be used')
    parser.add_argument('--max_dist_um', type=int, default=10,
        help='Define the maximum distance (in um) to be considered for the AP metric')
    parser.add_argument('--voxelsize', nargs=3, type=float, default=[0.7188, 0.7188, 0.7188],
        help='Define the voxel size (Z, Y, X) to use')
    parser.add_argument('--stats', type=bool, default=False,
        help='Define if statistical tests should be performed')
    args, _ = parser.parse_known_args()

    if isinstance(args.metric, str): # if only one metric is passed, convert to list
        args.metric = [args.metric]

    return args

if __name__ == '__main__':
    args = parse_args()
    main(**vars(args))