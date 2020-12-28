import os
import timeit
from urllib.parse import urljoin

import h5py
from astropy.samp import SAMPIntegratedClient

import pytest
from lib import SDSSCubeReader as h5r
from lib import photometry as cu

H5PATH = "../../SDSS_cube.h5"

@pytest.mark.incremental
class TestH5Reader:

    def setup_method(self, test_method):
        self.h5_file = h5py.File(H5PATH, 'r', track_order=True)
        self.cube_utils = cu.Photometry("../../config/SDSS_Bands",
                                       "../../config/ccd_gain.tsv",
                                       "../../config/ccd_dark_variance.tsv")
        self.resolution = 0

    def test_get_spectral_cube(self):
        self.reader = h5r.SDSSCubeReader(self.h5_file, self.cube_utils)
        data = self.reader.get_spectral_cube_from_orig_for_res(0)
        assert data.shape[1] == 1

    def test_write_VO_table(self):
        self.reader = h5r.SDSSCubeReader(self.h5_file, self.cube_utils)
        self.output_path = "output.xml"
        self.reader.get_spectral_cube_from_orig_for_res(0)
        self.reader.write_VOTable(self.output_path)
        #self.send_samp("table.load.votable")
        assert True

    def test_write_FITS(self):
        self.reader = h5r.SDSSCubeReader(self.h5_file, self.cube_utils)
        self.output_path = "output.fits"
        self.reader.get_spectral_cube_from_orig_for_res(self.resolution)
        self.reader.write_FITS(self.output_path)
        #self.send_samp("table.load.fits")
        assert True

    def test_write_FITS_zoomed(self):
        self.reader = h5r.SDSSCubeReader(self.h5_file, self.cube_utils)
        self.resolution = 3
        self.test_write_FITS()

    def test_write_FITS_from_dense(self):
        self.output_path = "output.fits"
        start_time = timeit.default_timer()
        self.reader = h5r.SDSSCubeReader(self.h5_file, self.cube_utils)
        self.reader.get_spectral_cube_for_res(0)
        print(timeit.default_timer() - start_time)
        self.reader.write_FITS(self.output_path)
        #self.send_samp("table.load.fits")
        assert True

    def send_samp(self, message_type):
        client = SAMPIntegratedClient()
        client.connect()
        params = {"url": urljoin('file:', os.path.abspath(self.output_path)), "name": "SDSS Cube"}
        print(params["url"])
        message = {"samp.mtype": message_type, "samp.params": params}
        client.notify_all(message)
