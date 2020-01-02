import os
import os.path as osp
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from pegc import constants
from pegc.data_prep.utils import get_subject_and_experiment_type_from_filename, to_one_hot_encoding
import putemg_features


def _denoise_filter_single_subject(src_hdf5_path: str, filtered_dara_dir_path: str) -> None:
    basename = osp.basename(src_hdf5_path)
    filename = osp.splitext(basename)[0]

    # read raw putEMG data file and run filter
    df: pd.DataFrame = pd.read_hdf(src_hdf5_path)
    putemg_features.biolab_utilities.apply_filter(df)

    # save filtered data to designated folder with prefix filtered_
    df.to_hdf(osp.join(filtered_dara_dir_path, f'{filename}_filtered.hdf5'),
              'data', format='table', mode='w', complevel=5)


def _calculate_single_subject_features(denoised_hdf5_path: str, features_dir_path: str) -> None:
    basename = os.path.basename(denoised_hdf5_path)
    filename = os.path.splitext(basename)[0]

    # for filtered data file run feature extraction, use xml with limited feature set
    ft: pd.DataFrame = putemg_features.features_from_xml('./sandbox/features_shallow_learn.xml', denoised_hdf5_path)  # Move this to some constant/parmater

    # save extracted features file to designated folder with features_filtered_ prefix
    ft.to_hdf(osp.join(features_dir_path, f'{filename}_filtered_features.hdf5'),
              'data', format='table', mode='w', complevel=5)
# Note: currently we operate only on row signals, but perhaps it will also be handy later.


def _process_single_filtered_hdf5(raw_filtered_data_dir: str, filename: str, processed_data_dir: str,
                                  window_size=constants.WINDOW_SIZE, window_stride=constants.WINDOW_STRIDE) -> None:
    df = pd.read_hdf(osp.join(raw_filtered_data_dir, filename))
    features_columns = [col for col in df.columns if col.startswith('EMG_')]
    df = df[df['TRAJ_GT'] != -1]  # drop invalid gestures

    label_diff = np.diff(df['TRAJ_GT'])
    gestures_boundaries = np.concatenate([[0], np.argwhere(label_diff != 0).flatten() + 1, [len(label_diff) - 1]])
    emg_features = df[features_columns].values

    X = []
    y = []
    gestures_labels = df['TRAJ_GT'].values
    for gesture_start_idx, gesture_end_idx in zip(gestures_boundaries[0: -1], gestures_boundaries[1:]):
        cur_gesture_features = emg_features[gesture_start_idx: gesture_end_idx]  # Note: check for off by 1 error
        cur_gesture_label = gestures_labels[gesture_start_idx]
        assert np.all(gestures_labels[gesture_start_idx: gesture_end_idx] == cur_gesture_label)

        for window_start_idx in range(0, gesture_end_idx - gesture_start_idx - 1,
                                      window_stride):  # TODO: to numpy (maybe)
            window_features = cur_gesture_features[window_start_idx: window_start_idx + window_size]
            if len(window_features) != window_size:  # ignore the "not full" features at gesture end
                continue
            X.append(cur_gesture_features[window_start_idx: window_start_idx + window_size])
            y.append(cur_gesture_label)

    X = np.array(X)
    y = np.array(y)

    # Note/Important: the original dataset has a gap (no gestures with classes 4/5) which is addressed by simply
    # subtracting 2 from classes above 5.
    y[y >= 6] -= 2
    y = to_one_hot_encoding(y, constants.NB_DATASET_CLASSES)

    # save results
    sub_id, exp_type = get_subject_and_experiment_type_from_filename(filename)
    sub_dir_path = osp.join(processed_data_dir, sub_id)

    np.save(osp.join(sub_dir_path, f'{exp_type}_X.npy'), X)
    np.save(osp.join(sub_dir_path, f'{exp_type}_y.npy'), y)


def _prepare_single_subject_splits(processed_data_dir: str, processed_data_splits_dir: str,
                                   subject_dir: str) -> None:
    possible_splits = {'split_0': {'train': ('sequential', 'repeats_short'), 'test': 'repeats_long'},
                   'split_1': {'train': ('sequential', 'repeats_long'), 'test': 'repeats_short'},
                   'split_2': {'train': ('repeats_short', 'repeats_long'), 'test': 'sequential'}}
    subject_dir_path = osp.join(processed_data_dir, subject_dir)

    # load all data once
    experiments_data = {}
    for exp_type in ('repeats_long', 'repeats_short', 'sequential'):
        X = np.load(osp.join(subject_dir_path, f'{exp_type}_X.npy'))
        y = np.load(osp.join(subject_dir_path, f'{exp_type}_y.npy'))
        experiments_data[exp_type] = {'X': X, 'y': y}

    splits_dst_dir_path = osp.join(processed_data_splits_dir, subject_dir)
    os.makedirs(splits_dst_dir_path, exist_ok=True)
    for split_name, train_test_info in possible_splits.items():
        X_train = np.concatenate([experiments_data[train_test_info['train'][0]]['X'],
                                  experiments_data[train_test_info['train'][1]]['X']])
        y_train = np.concatenate([experiments_data[train_test_info['train'][0]]['y'],
                                  experiments_data[train_test_info['train'][1]]['y']])

        X_test = experiments_data[train_test_info['test']]['X']
        y_test = experiments_data[train_test_info['test']]['y']

        # Note: features must be "unrolled" to 2 dims to apply StandardScaler and then "rolled" back
        # to the original.
        orig_X_train_shape = X_train.shape
        orig_X_test_shape = X_test.shape
        X_train = X_train.reshape((X_train.shape[0] * X_train.shape[1], X_train.shape[2]))
        X_test = X_test.reshape((X_test.shape[0] * X_test.shape[1], X_test.shape[2]))
        # Note/potential problem/TODO: due to how te samples are constructed (with moving window), some of
        # the signals occur 2 times while the ones ate the beginning/end of a gesture only once – which
        # propably slightly affects results after scaling.
        sc = StandardScaler().fit(X_train)
        X_train = sc.transform(X_train)
        X_test = sc.transform(X_test)
        X_train = X_train.reshape(orig_X_train_shape)
        X_test = X_test.reshape(orig_X_test_shape)

        # Note/TODO: add shuffle? save scaler?

        cur_split_dst_dir_path = osp.join(splits_dst_dir_path, split_name)
        os.makedirs(cur_split_dst_dir_path, exist_ok=True)

        np.save(osp.join(cur_split_dst_dir_path, 'X_train.npy'), X_train)
        np.save(osp.join(cur_split_dst_dir_path, 'y_train.npy'), y_train)
        np.save(osp.join(cur_split_dst_dir_path, 'X_test.npy'), X_test)
        np.save(osp.join(cur_split_dst_dir_path, 'y_test.npy'), y_test)


def _check_examination_splits_shapes_validity(examination_dir_path: str) -> Tuple[int, bool]:
    examination_id = int(osp.basename(examination_dir_path))

    shapes_valid = True
    for i in range(3):
        split_dir_path = osp.join(examination_dir_path, f'split_{i}')
        X_train = np.load(osp.join(split_dir_path, 'X_train.npy'))
        X_test = np.load(osp.join(split_dir_path, 'X_test.npy'))
        y_train = np.load(osp.join(split_dir_path, 'y_train.npy'))
        y_test = np.load(osp.join(split_dir_path, 'y_test.npy'))
        if (len(X_train) != len(y_train) or len(X_test) != len(y_test) or
                y_train.shape[1] != constants.NB_DATASET_CLASSES or y_test.shape[1] != constants.NB_DATASET_CLASSES):
            shapes_valid = False
            break

    return examination_id, shapes_valid
