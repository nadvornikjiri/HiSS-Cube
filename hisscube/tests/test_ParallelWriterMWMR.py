import pytest
import h5py

from hisscube.ParallelWriterMWMRService import ParallelWriterMWMRService

H5PATH = "../../results/SDSS_cube_parallel.h5"
INPUT_PATH = "../../data/raw/galaxy_small"


@pytest.fixture(scope="session", autouse=False)
def truncate_test_file(request):
    h5path = H5PATH
    f = h5py.File(h5path, 'w', libver="latest")  # create + truncate file
    f.close()


@pytest.mark.incremental
class TestH5ParallelWriter:

    def test_reingest_fits_tables(self):
        writer = ParallelWriterMWMRService(h5_path=H5PATH)
        fits_image_path = "%s/images" % INPUT_PATH
        fits_spectra_path = "%s/spectra" % INPUT_PATH
        pattern = "*.fits"
        writer.open_h5_file()
        writer.reingest_fits_tables(fits_image_path, fits_spectra_path, pattern, pattern)
        assert True

    def test_ingest_metadata(self):
        writer = ParallelWriterMWMRService(h5_path=H5PATH)
        writer.open_h5_file()
        writer.ingest_metadata()
        assert True



