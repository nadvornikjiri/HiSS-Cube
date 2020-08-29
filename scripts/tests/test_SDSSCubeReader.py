import h5py
import pytest
from scripts import photometry as cu
from scripts import SDSSCubeReader as h5r
import numpy as np

H5PATH = "../../data/SDSS_cube.h5"

class TestH5Reader:

    def setup_method(self, test_method):
        self.h5_file = h5py.File(H5PATH, 'r', track_order=True)
        self.cube_utils = cu.CubeUtils("../../config/SDSS_Bands",
                                       "../../config/ccd_gain.tsv",
                                       "../../config/ccd_dark_variance.tsv")
        self.reader = h5r.SDSSCubeReader(self.h5_file, self.cube_utils)

    def test_get_spectral_cube(self):
        data = self.reader.get_spectral_cube_for_res(0)
        assert data.shape[1] == 5

    def test_write_VOTable(self):
        self.reader.get_spectral_cube_for_res(0)
        self.reader.write_VOTable("output.xml")
