MEM_REQ_PER_PROCESS = 1.6e9  # It is rather much lower for second stage, might need adjustment for denoising.
DATASET_FEATURES_SHAPE = (24, 1024)  # Stored in reverse on drive, but it is adjusted for pytorch during loading.
NB_DATASET_CLASSES = 8
RANDOM_SEED = 42
