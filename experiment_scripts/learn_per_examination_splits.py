import os
import os.path as osp
from typing import Optional, Sequence, Dict

import click
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pegc.training import train_loop
from pegc.training.utils import load_json


def classification_report(cm: np.array, class_labels: Optional[Sequence[str]] = None) -> Dict[str, float]:
    assert cm.shape[0] == cm.shape[1] and len(class_labels) == len(cm) if class_labels is not None else True
    tp = np.diag(cm)
    fp = np.sum(cm, axis=0) - tp
    fn = np.sum(cm, axis=1) - tp
    nb_classes = len(cm)
    tn = []
    for i in range(nb_classes):
        tmp = np.delete(cm, i, 0)
        tmp = np.delete(tmp, i, 1)
        tn.append(np.sum(tmp))
    tn = np.array(tn)
    # assert np.all() sanity check for sum of all of the above for respective classes equal to samples in
    # ground truth would go here – but we don't have the ground truth in this scope.
    precision = tp / (tp + fp)
    precision = np.nan_to_num(precision)
    recall = tp / (tp + fn)
    recall = np.nan_to_num(recall)
    f1 = 2 * precision * recall / (precision + recall)

    if class_labels is None:
        class_labels = [f'class_{i}' for i in range(len(cm))]
    label_tp_map = {f'tp_{label}': tp for label, tp in zip(class_labels, tp)}
    label_tn_map = {f'tn_{label}': tn for label, tn in zip(class_labels, tn)}
    label_fp_map = {f'fp_{label}': fp for label, fp in zip(class_labels, fp)}
    label_fn_map = {f'fn_{label}': fn for label, fn in zip(class_labels, fn)}
    label_precision_map = {f'precision_{label}': prec for label, prec in zip(class_labels, precision)}
    label_recall_map = {f'recall_{label}': rec for label, rec in zip(class_labels, recall)}
    label_f1_map = {f'f1_{label}': f1 for label, f1 in zip(class_labels, f1)}

    results = {**label_tp_map, ** label_tn_map, **label_fp_map, **label_fn_map, **label_f1_map,
               **label_precision_map, **label_recall_map}
    return results


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
    gestures_classes = ['idle', 'fist', 'flexion', 'extension', 'pinch_index', 'pinch_middle',
                        'pinch_ring', 'pinch_small']
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
            cm = np.array(test_eval_metrics['cm'])
            clf_report = classification_report(cm, gestures_classes)
            final_eval_results.append([examination_id, split_name, train_metrics[-1]['val_loss'], train_metrics[-1]['val_acc'],
                                       test_eval_metrics['val_loss'], test_eval_metrics['val_acc'], test_eval_metrics['cm']] +
                                       [clf_report[k] for k in sorted(clf_report)])

    res_df = pd.DataFrame(final_eval_results, columns=['examination_id', 'split_name', 'val_loss', 'val_acc',
                                                       'test_loss', 'test_acc', 'test_cm'] + sorted(list(clf_report.keys())))
    res_df.to_csv(osp.join(results_output_dir_path, 'final_evaluations_aggregated.csv'), index=False, float_format='%.4f')

    # TODO: some (more) plots?
    plt.boxplot([res_df['test_acc']])
    plt.axhline(y=0.59, label='majority class')  # Baseline (mean majority class share – value is estimated, compute it precisely later).
    # plt.title('Test set accs')
    plt.legend(loc='upper right')
    plt.ylabel('test set acc')
    plt.xticks([])
    plt.savefig(osp.join(results_output_dir_path, 'test_accs_boxplot.png'))


if __name__ == '__main__':
    run_experiment()
