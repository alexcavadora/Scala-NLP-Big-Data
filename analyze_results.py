import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
path = "results/3_final"
def load_all_results(results_dir=f'{path}', sentiment_levels=None):
    """Load all metrics_summary.csv files from results directory"""
    results = []

    for model_dir in Path(results_dir).iterdir():
        if model_dir.is_dir():
            metrics_file = model_dir / 'metrics_summary.csv'
            if metrics_file.exists():
                df = pd.read_csv(metrics_file)
                model_name = model_dir.name
                df['model_task'] = model_name

                # Parse model and task
                parts = model_name.rsplit('_', 1)
                if len(parts) == 2:
                    df['model'] = parts[0]
                    task = parts[1]
                    if task == 'sentiment' and sentiment_levels is not None:
                        task = f'sentiment({sentiment_levels})'
                    df['task'] = task
                else:
                    df['model'] = model_name
                    df['task'] = 'unknown'

                results.append(df)

    if not results:
        print("No results found!")
        return None

    return pd.concat(results, ignore_index=True)

def plot_model_comparison(df, metric='accuracy', save_dir=f'{path}/analysis'):
    """Plot comparison of models for a specific metric, showing train vs test scores."""
    os.makedirs(save_dir, exist_ok=True)

    for task in df['task'].unique():
        task_df = df[df['task'] == task]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), gridspec_kw={'width_ratios': [2.5, 1]})

        # Train vs Test comparison
        test_df = task_df[task_df['split'] == 'test'].sort_values(metric, ascending=False)
        if test_df.empty:
            continue

        models = test_df['model'].values
        test_scores = test_df[metric].values

        train_scores = []
        for model in models:
            train_score_val = task_df[(task_df['model'] == model) & (task_df['split'] == 'train')][metric].values
            train_scores.append(train_score_val[0] if len(train_score_val) > 0 else 0)

        x = np.arange(len(models))
        width = 0.35

        ax1.bar(x - width/2, train_scores, width, label='Train', alpha=0.8)
        ax1.bar(x + width/2, test_scores, width, label='Test', alpha=0.8)
        ax1.set_xlabel('Model', fontsize=12)
        ax1.set_ylabel(metric.capitalize(), fontsize=12)
        ax1.set_title(f'{metric.capitalize()} Comparison (Train vs Test) - {task.capitalize()} Task', fontsize=14)
        ax1.set_xticks(x)
        ax1.set_xticklabels(models, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.4)

        # Overfitting analysis
        overfit = np.array(train_scores) - np.array(test_scores)

        overfit_df = pd.DataFrame({'model': models, 'gap': overfit}).sort_values('gap', ascending=False)

        colors = ['red' if x > 0.1 else 'green' for x in overfit_df['gap']]

        sns.barplot(x='model', y='gap', data=overfit_df, palette=colors, ax=ax2, alpha=0.7)
        ax2.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
        ax2.axhline(y=0.1, color='red', linestyle='--', linewidth=0.8, alpha=0.5)
        ax2.set_xlabel('Model', fontsize=12)
        ax2.set_ylabel('Train - Test Gap', fontsize=12)
        ax2.set_title('Overfitting Analysis', fontsize=14)
        ax2.tick_params(axis='x', rotation=45, labelsize=10)
        ax2.grid(axis='y', alpha=0.4)

        fig.tight_layout(pad=3.0)
        plt.savefig(f'{save_dir}/{metric}_comparison_{task}.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved: {metric}_comparison_{task}.png")


def plot_all_metrics_heatmap(df, save_dir='results/analysis'):
    """Create heatmap of all metrics for all models for train and test splits"""
    os.makedirs(save_dir, exist_ok=True)

    metrics = ['accuracy', 'precision', 'recall', 'f1']

    for task in df['task'].unique():
        for split in ['train', 'test']:
            task_df = df[(df['task'] == task) & (df['split'] == split)]

            if task_df.empty:
                continue

            pivot_df = task_df.pivot_table(
                values=metrics,
                index='model',
                aggfunc='first'
            )

            plt.figure(figsize=(10, 6))
            sns.heatmap(pivot_df, annot=True, fmt='.4f', cmap='YlOrRd',
                       cbar_kws={'label': 'Score'})
            plt.title(f'All Metrics Heatmap - {task.capitalize()} Task ({split.capitalize()} Set)')
            plt.xlabel('Metric')
            plt.ylabel('Model')
            plt.tight_layout()
            plt.savefig(f'{save_dir}/metrics_heatmap_{task}_{split}.png', dpi=300, bbox_inches='tight')
            plt.close()

            print(f"Saved: metrics_heatmap_{task}_{split}.png")

def generate_summary_table(df, save_dir='results/analysis'):
    """Generate summary table with best models for each metric for train and test splits"""
    os.makedirs(save_dir, exist_ok=True)

    metrics = ['accuracy', 'precision', 'recall', 'f1']

    for task in df['task'].unique():
        for split in ['train', 'test']:
            task_df = df[(df['task'] == task) & (df['split'] == split)]

            if task_df.empty:
                continue

            print(f"\n{'='*60}")
            print(f"SUMMARY - {task.upper()} TASK ({split.upper()} SET)")
            print(f"{'='*60}\n")

            summary_data = []

            for metric in metrics:
                if metric not in task_df.columns: continue
                best_row = task_df.loc[task_df[metric].idxmax()]
                worst_row = task_df.loc[task_df[metric].idxmin()]

                summary_data.append({
                    'Metric': metric.capitalize(),
                    'Best Model': best_row['model'],
                    'Best Score': f"{best_row[metric]:.4f}",
                    'Worst Model': worst_row['model'],
                    'Worst Score': f"{worst_row[metric]:.4f}",
                    'Range': f"{best_row[metric] - worst_row[metric]:.4f}"
                })

                print(f"{metric.capitalize():12} - Best: {best_row['model']:20} ({best_row[metric]:.4f})")

            summary_df = pd.DataFrame(summary_data)
            summary_df.to_csv(f'{save_dir}/summary_{task}_{split}.csv', index=False)
            print(f"\nSummary saved to: summary_{task}_{split}.csv")

            # Full leaderboard
            print(f"\n{'='*60}")
            print(f"LEADERBOARD - {task.upper()} TASK ({split.upper()} SET)")
            print(f"{'='*60}\n")
            leaderboard = task_df.sort_values('f1', ascending=False)[
                ['model', 'accuracy', 'precision', 'recall', 'f1']
            ]
            print(leaderboard.to_string(index=False))
            leaderboard.to_csv(f'{save_dir}/leaderboard_{task}_{split}.csv', index=False)

def main():
    print("Review Classifier - Results Analysis")
    print("=" * 60)

    # Configure sentiment levels (e.g., 3 or 5)
    # Set to None to use 'sentiment' as the default task name
    SENTIMENT_LEVELS = 3

    # Load all results
    df = load_all_results(sentiment_levels=SENTIMENT_LEVELS)

    if df is None:
        return

    print(f"\nLoaded results from {len(df['model_task'].unique())} experiments")
    print(f"Tasks: {', '.join(df['task'].unique())}")
    print(f"Models: {', '.join(df['model'].unique())}")

    # Create analysis directory
    analysis_dir = f'{path}/analysis'
    os.makedirs(analysis_dir, exist_ok=True)

    # Generate all plots
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1']

    print("\nGenerating comparison plots...")
    for metric in metrics_to_plot:
        plot_model_comparison(df, metric, save_dir=analysis_dir)

    print("\nGenerating heatmaps...")
    plot_all_metrics_heatmap(df, save_dir=analysis_dir)

    print("\nGenerating summary tables...")
    generate_summary_table(df, save_dir=analysis_dir)

    print("\n" + "=" * 60)
    print(f"Analysis complete! Check {analysis_dir}/ directory")
    print("=" * 60)

if __name__ == '__main__':
    main()
