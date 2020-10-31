import h5py
from scripts import photometry as cu
from scripts import FITSCubeReader as fits_reader


class TestFITSCubeReader:

    def setup_method(self, test_method):
        self.cube_utils = cu.CubeUtils("../../config/SDSS_Bands",
                                       "../../config/ccd_gain.tsv",
                                       "../../config/ccd_dark_variance.tsv")
        spectra_path = "../../data/galaxy_small/spectra"
        image_path = "../../data/galaxy_small/images"
        self.reader = fits_reader.FITSCubeReader(spectra_path, image_path, self.cube_utils)
        self.resolution = 0

    def test_get_spectral_cube(self):
        data = self.reader.get_spectral_cube_for_res(self.resolution)
        assert data.shape[1] == 1

    def test_write_FITS(self):
        self.output_path = "output.fits"
        self.reader.get_spectral_cube_for_res(self.resolution)
        self.reader.write_FITS(self.output_path)
        self.send_samp("table.load.fits")
        assert True
