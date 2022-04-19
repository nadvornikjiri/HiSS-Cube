import sys

sys.path.append('../')
import warnings
from pathlib import Path

from astropy.utils.exceptions import AstropyWarning
from tqdm.auto import tqdm
from hisscube.Photometry import Photometry
from hisscube.Writer import Writer
from hisscube.MLProcessor import MLProcessor
import h5py
from timeit import default_timer as timer
import subprocess
import os

H5PATH = "../data/processed/galaxy_small.h5"
image_folder = "../data/raw/galaxy_small/images"
spectra_folder = "../data/raw/galaxy_small/spectra"
image_pattern = "*.fits"
spectra_pattern = "*.fits"

warnings.simplefilter('ignore', category=AstropyWarning)

cube_utils = Photometry("../config/SDSS_Bands",
                        "../config/ccd_gain.tsv",
                        "../config/ccd_dark_variance.tsv")

with h5py.File(H5PATH, 'w') as h5_file:  # truncate file
    start1 = timer()
    writer = Writer(h5_file)
    image_paths = list(Path(image_folder).rglob(image_pattern))
    for image in tqdm(image_paths, desc="Images completed: "):
        writer.ingest_image(image)
    start2 = timer()
    print("Image ingestion time: %.2f" % (start2 - start1))
    spectra_paths = list(Path(spectra_folder).rglob(spectra_pattern))
    for spectrum in tqdm(spectra_paths, desc="Spectra Progress: "):
        writer.ingest_spectrum(spectrum)
    start1 = start2
    start2 = timer()
    print("Spectra time: %.2fs" % (start2 - start1))

    writer.add_image_refs(h5_file)
    writer.create_dense_cube()
    start1 = start2
    start2 = timer()
    print("Created dense cube + image_refs: %.2fs" % (start2 - start1))

    ml_writer = MLProcessor(h5_file)
    ml_writer.create_3d_cube()
    start1 = start2
    start2 = timer()
    print("Created 3d cube: %.2fs" % (start2 - start1))

process = subprocess.Popen("cat /proc/%s/io" % os.getpid(), shell=True)
process.wait()
