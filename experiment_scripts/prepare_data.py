import os
import os.path as osp
import warnings

import numpy as np
from tqdm import tqdm
import psutil
from joblib import Parallel, delayed
import click

from pegc import constants
from pegc.data_prep.processing_jobs import _prepare_single_subject_splits, _process_single_filtered_hdf5, \
    _denoise_filter_single_subject, _check_examination_splits_shapes_validity


@click.command()
@click.argument('orig_data_dir_path', type=click.Path(exists=True, file_okay=False))
@click.argument('raw_filtered_data_path', type=click.Path(file_okay=False))
@click.argument('processed_data_dir', type=click.Path(file_okay=False))
@click.argument('processed_data_splits_dir', type=click.Path(file_okay=False))
@click.option('--window_size', default=1024, type=int)
@click.option('--window_stride', default=512, type=int)
@click.option('--clean_intermediate_steps', default=True, type=bool)
def prepare_data(orig_data_dir_path: str, raw_filtered_data_path: str, processed_data_dir: str,
                 processed_data_splits_dir: str, window_size: int = 1024, window_stride: int = 512,
                 clean_intermediate_steps: bool = True) -> None:
    nb_workers = max(int(np.floor(psutil.virtual_memory()[1] / constants.MEM_REQ_PER_PROCESS)), 1)
    nb_workers = min(nb_workers, os.cpu_count())  # Limit workers so that there is no more of them than actual available cores.

    # Denoise/filter data
    os.makedirs(raw_filtered_data_path, exist_ok=True)
    Parallel(n_jobs=nb_workers)(delayed(_denoise_filter_single_subject)(osp.join(orig_data_dir_path, filename),
                                                                        raw_filtered_data_path)
                                for filename in tqdm(os.listdir(orig_data_dir_path)))

    # Group and preprocess raw (but filtered/denoised) signals for each subject.
    os.makedirs(processed_data_dir, exist_ok=True)
    Parallel(n_jobs=nb_workers)(delayed(_process_single_filtered_hdf5)(raw_filtered_data_path, filename,
                                                                       processed_data_dir, window_size, window_stride,
                                                                       clean_intermediate_steps)
                                for filename in tqdm(os.listdir(raw_filtered_data_path)))
    if clean_intermediate_steps:
        os.rmdir(raw_filtered_data_path)

    # Create inter-subject train/test splits (according to the original dataset authors methodology).
    os.makedirs(processed_data_splits_dir, exist_ok=True)
    Parallel(n_jobs=nb_workers)(delayed(_prepare_single_subject_splits)(processed_data_dir,
                                                                        processed_data_splits_dir,
                                                                        subject_dir, clean_intermediate_steps)
                                for subject_dir in tqdm(os.listdir(processed_data_dir)))
    if clean_intermediate_steps:
        os.rmdir(processed_data_dir)

    # Optional/debug: check if in all generated splits lengths of X/y matches
    validation_results = Parallel(n_jobs=nb_workers)(delayed(_check_examination_splits_shapes_validity)(
        osp.join(processed_data_splits_dir, subject_dir))
        for subject_dir in tqdm(os.listdir(processed_data_splits_dir)))
    examinations_with_invalid_shapes = {exam_id for exam_id, shapes_are_valid in validation_results
                                        if not shapes_are_valid}
    if examinations_with_invalid_shapes:
        warnings.warn(f'Shapes of data in splits of examinations with ids'
                      f' {examinations_with_invalid_shapes} are incorrect!')


if __name__ == '__main__':
    prepare_data()
