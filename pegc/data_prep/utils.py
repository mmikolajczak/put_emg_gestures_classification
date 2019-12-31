import os
import os.path as osp
import re
from typing import Set, Tuple

import numpy as np


def get_subject_and_experiment_type_from_filename(filename: str) -> Tuple[str, str]:
    pat = '^emg_gestures-(\d+)-(\w+)-.*?\.hdf5$'
    match = re.match(pat, filename)
    subject_id, experiment_type = match.group(1), match.group(2)
    return subject_id, experiment_type


def get_subjects_ids(raw_filtered_data_dir: str) -> Set[str]:
    subjects_ids = set(get_subject_and_experiment_type_from_filename(f)[0]
                       for f in os.listdir(raw_filtered_data_dir))
    return subjects_ids


def prepare_dir_tree(processed_data_dir: str, subjects_ids: Set[str]) -> None:
    os.makedirs(processed_data_dir, exist_ok=True)
    for id_ in subjects_ids:
        id_subdir_path = osp.join(processed_data_dir, id_)
        os.makedirs(id_subdir_path, exist_ok=True)


def to_one_hot_encoding(arr_1d: np.array, nb_classes: int) -> np.array:
    nb_rows = len(arr_1d)

    ohe_arr = np.zeros(nb_rows * nb_classes)
    ones_idxs = (np.arange(nb_rows) * nb_classes + arr_1d).astype(np.uint32)
    ohe_arr[ones_idxs] = 1
    ohe_arr = ohe_arr.reshape((nb_rows, nb_classes))
    return ohe_arr
