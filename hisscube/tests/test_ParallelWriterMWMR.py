
import pytest
import h5py

from hisscube.ParallelWriterMWMR import ParallelWriterMWMR

H5PATH = "../../data/processed/SDSS_cube_parallel.h5"
INPUT_PATH = "../../data/raw/galaxy_small"

@pytest.fixture(scope="session", autouse=False)
def truncate_test_file(request):
    h5path = H5PATH
    f = h5py.File(h5path, 'w', libver="latest")  # create + truncate file
    f.close()

@pytest.mark.incremental
class TestH5ParallelWriter:

    def setup_method(self, test_method):
        self.h5_file = h5py.File(H5PATH, 'r+', libver="latest")

    def teardown_method(self, test_method):
        self.h5_file.close()

    @pytest.mark.usefixtures("truncate_test_file")
    def test_ingest_metadata(self):
        writer = ParallelWriterMWMR(h5_file=self.h5_file)
        fits_image_path = "%s/images" % INPUT_PATH
        fits_spectra_path = "%s/spectra" % INPUT_PATH
        writer.ingest_metadata(fits_image_path, fits_spectra_path)
        assert True
