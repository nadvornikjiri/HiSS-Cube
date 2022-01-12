import sys
sys.path.append('../')
from pathlib import Path
import logging
logging.basicConfig(level = logging.INFO)

from tqdm import tqdm

from hisscube.Writer import Writer

H5PATH = "../results/SDSS_cube_parallel.h5"
FITS_IMAGE_PATH = "../data/raw/galaxy_small/images"
ITERATIONS = 3


def run_test(i, no_attrs, no_datasets, log_name):
    print("Running test for %s, iteration %d." %(log_name, i))
    writer = Writer(h5_path=H5PATH, timings_log="logs/%s_%d.csv" % (log_name, i))
    writer.open_h5_file_serial(truncate=True)
    image_pattern, spectra_pattern = writer.get_path_patterns()
    writer.write_images_metadata(FITS_IMAGE_PATH, image_pattern,no_attrs=no_attrs, no_datasets=no_datasets)
    writer.close_h5_file()


for i in range(ITERATIONS):
    run_test(i, no_attrs=True, no_datasets=True, log_name="groups_only")
    run_test(i, no_attrs=True, no_datasets=False, log_name="groups_datasets")
    run_test(i, no_attrs=False, no_datasets=False, log_name="groups_attrs_datasets")





