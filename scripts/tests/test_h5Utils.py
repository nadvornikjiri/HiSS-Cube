import fitsio
from scripts import h5Utils as h5u
import h5py
import pytest


@pytest.fixture(scope="session", autouse=True)
def do_something(request):
    h5path = "SDSS_cube.h5"
    h5py.File(h5path, 'w')  # create + truncate file


class TestCubeUtils:

    def setup_method(self, test_method):
        self.h5path = "SDSS_cube.h5"

    def test_add_image(self):
        test_path = "../../data/images/301/4797/1/frame-g-004797-1-0019.fits.bz2"
        with fitsio.FITS(test_path) as f:
            writer = h5u.SDSSCubeWriter(self.h5path, test_path,
                                        "../../config/SDSS_Bands",
                                        "../../config/ccd_gain.tsv",
                                        "../../config/ccd_dark_variance.tsv")
            h5_datasets = writer.ingest_image()
            assert len(h5_datasets) == 4
            writer.close_hdf5()

    def test_add_spectrum(self):
        test_path = "../../data/spectra/spec-4500-55543-0331.fits"
        with fitsio.FITS(test_path) as f:
            writer = h5u.SDSSCubeWriter(self.h5path, test_path,
                                        "../../config/SDSS_Bands",
                                        "../../config/ccd_gain.tsv",
                                        "../../config/ccd_dark_variance.tsv")
            h5_datasets = writer.ingest_spectrum()
            assert len(h5_datasets) == 5
            writer.close_hdf5()
