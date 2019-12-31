WINDOW_SIZE = 1024
WINDOW_STRIDE = 512
MEM_REQ_PER_PROCESS = 1.6e9  # it is rather much lower for first stage

# Data must by first filtered using code from the original authors repository.
# (Thanks to the Windows this files went to hell so welp, re-generation will be needed).
# Wygeneruj to na kompie za ścianą, spytać Dominika ile tam miejsca jest przy okazji, będzie szybciej niż
# potem przepychać to przez sieć wszystko.
# Zobaczyć czy kompresja tych wynikowych danych jakoś pomaga może.
ORIG_DATA_DIR = '/media/ja/CCTV_nagrania/mkm_archive/put_emg/data/orig_hdf5'
RAW_FILTERED_DATA_DIR = '/media/ja/CCTV_nagrania/mkm_archive/put_emg/data/filtered_data/'
PROCESSED_DATA_DIR = (f'/media/ja/CCTV_nagrania/mkm_archive/put_emg/data/raw_filtered_data_subjects_'
                      f'split_window_size_{WINDOW_SIZE}_window_stride_{WINDOW_STRIDE}')
PROCESSED_DATA_SPLITS_DIR = (f'/media/ja/CCTV_nagrania/mkm_archive/put_emg/data/raw_filtered_data_subjects_'
                             f'split_window_size_{WINDOW_SIZE}_window_stride_{WINDOW_STRIDE}_cv_splits_standarized')
