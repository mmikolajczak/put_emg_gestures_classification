import os
import re
from typing import Set, Tuple

import numpy as np


def get_experiment_metadata_from_filename(filename: str) -> Tuple[str, str, str]:
    pat = r'^emg_gestures-(\d+)-(\w+)-(\d{4}-\d{2}-\d{2}).*?\.hdf5$'
    match = re.match(pat, filename)
    subject_id, experiment_type, experiment_date = match.group(1), match.group(2), match.group(3)
    return subject_id, experiment_type, experiment_date


def get_subjects_ids(raw_filtered_data_dir: str) -> Set[str]:
    subjects_ids = set(get_experiment_metadata_from_filename(f)[0]
                       for f in os.listdir(raw_filtered_data_dir))
    return subjects_ids


def to_one_hot_encoding(arr_1d: np.array, nb_classes: int) -> np.array:
    nb_rows = len(arr_1d)

    ohe_arr = np.zeros(nb_rows * nb_classes)
    ones_idxs = (np.arange(nb_rows) * nb_classes + arr_1d).astype(np.uint32)
    ohe_arr[ones_idxs] = 1
    ohe_arr = ohe_arr.reshape((nb_rows, nb_classes))
    return ohe_arr
