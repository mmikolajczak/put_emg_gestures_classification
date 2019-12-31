import os
import os.path as osp

import numpy as np
from tqdm import tqdm
import psutil
from joblib import Parallel, delayed

from pegc.data_prep import constants
from pegc.data_prep.processing_jobs import _prepare_single_subject_splits, _process_single_filtered_hdf5, \
    _denoise_filter_single_subject
from pegc.data_prep.utils import get_subjects_ids, prepare_dir_tree


def prepare_data():
    nb_workers = max(int(np.floor(psutil.virtual_memory()[1] / constants.MEM_REQ_PER_PROCESS)), 1)

    # Denoise/filter data
    os.makedirs(constants.RAW_FILTERED_DATA_DIR, exist_ok=True)
    Parallel(n_jobs=nb_workers)(delayed(_denoise_filter_single_subject)(osp.join(constants.ORIG_DATA_DIR, filename),
                                                                        constants.RAW_FILTERED_DATA_DIR)
                                for filename in tqdm(os.listdir(constants.ORIG_DATA_DIR)))

    # Group and preprocess raw (but filtered/denoised) signals for each subject.
    put_emg_subjects_ids = get_subjects_ids(constants.RAW_FILTERED_DATA_DIR)
    prepare_dir_tree(constants.PROCESSED_DATA_DIR, put_emg_subjects_ids)
    Parallel(n_jobs=nb_workers)(delayed(_process_single_filtered_hdf5)(constants.RAW_FILTERED_DATA_DIR,
                                                                       filename,
                                                                       constants.PROCESSED_DATA_DIR)
                                for filename in tqdm(os.listdir(constants.RAW_FILTERED_DATA_DIR)))

    # Create inter-subject train/test splits (according to the original dataset authors methodology).
    os.makedirs(constants.PROCESSED_DATA_SPLITS_DIR, exist_ok=True)
    Parallel(n_jobs=nb_workers)(delayed(_prepare_single_subject_splits)(constants.PROCESSED_DATA_DIR,
                                                                        constants.PROCESSED_DATA_SPLITS_DIR,
                                                                        subject_dir)
                                for subject_dir in tqdm(os.listdir(constants.PROCESSED_DATA_DIR)))


if __name__ == '__main__':
    prepare_data()
