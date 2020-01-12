import os
import os.path as osp

import click
import pandas as pd

from pegc.training import train_loop
from pegc.training.utils import load_json


@click.command()
@click.argument('examinations_prepared_train_test_splits_dir_path', type=click.Path(file_okay=False, exists=True))
@click.argument('results_output_dir_path', type=click.Path(file_okay=False))
@click.argument('training_config_file_path', type=click.Path(dir_okay=False, exists=True))
def run_experiment(examinations_prepared_train_test_splits_dir_path: str,
                   results_output_dir_path: str, training_config_file_path: str) -> None:
    # Note: training config file is a dict in json format, while it's content should be arguments which will override
    # some of (or all) arguments of the train loop function.
    # Template with "default" train loop arguments/what can be changed is included in experiments scripts dir.
    possible_splits = {'split_0': {'train': ('sequential', 'repeats_short'), 'test': 'repeats_long'},
                       'split_1': {'train': ('sequential', 'repeats_long'), 'test': 'repeats_short'},
                       'split_2': {'train': ('repeats_short', 'repeats_long'), 'test': 'sequential'}}
    training_config = load_json(training_config_file_path)
    final_eval_results = []

    os.makedirs(results_output_dir_path, exist_ok=True)
    for examination_dir in os.listdir(examinations_prepared_train_test_splits_dir_path):
        examination_dir_path = osp.join(examinations_prepared_train_test_splits_dir_path, examination_dir)
        examination_id = examination_dir

        results_examination_dir_path = osp.join(results_output_dir_path, examination_dir)
        os.makedirs(results_examination_dir_path, exist_ok=True)
        for split_dir in os.listdir(examination_dir_path):
            split_dir_path = osp.join(examination_dir_path, split_dir)
            results_split_dir_path = osp.join(results_examination_dir_path, split_dir)
            train_loop(split_dir_path, results_split_dir_path, **training_config)

            # Add test evaluation results to summary csv.
            train_metrics = load_json(osp.join(results_split_dir_path, 'training_losses_and_metrics.json'))['epochs_stats']
            test_eval_metrics = load_json(osp.join(results_split_dir_path, 'test_set_stats.json'))
            split_name = f'train_{"_".join(possible_splits[split_dir]["train"])}_test_{possible_splits[split_dir]["test"]}'
            final_eval_results.append([examination_id, split_name, train_metrics[-1]['val_loss'], train_metrics[-1]['val_acc'],
                                       test_eval_metrics['val_loss'], test_eval_metrics['val_acc'], test_eval_metrics['cm']])

    res_df = pd.DataFrame(final_eval_results, columns=['examination_id', 'split_name', 'val_loss', 'val_acc',
                                                       'test_loss', 'test_acc', 'test_cm'])
    res_df.to_csv(osp.join(results_output_dir_path, 'final_evaluations_aggregated.csv'), index=False)
    # TODO: some plots?


if __name__ == '__main__':
    run_experiment()
