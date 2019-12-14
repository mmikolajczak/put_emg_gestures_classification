import os
import os.path as osp
import re
from typing import Set, Tuple


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
