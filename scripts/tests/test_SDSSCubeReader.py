import os
from urllib.parse import urljoin

import h5py
from astropy.samp import SAMPIntegratedClient

from scripts import SDSSCubeReader as h5r
from scripts import photometry as cu

H5PATH = "../../SDSS_cube_gzip9.h5"


class TestH5Reader:

    def setup_method(self, test_method):
        self.h5_file = h5py.File(H5PATH, 'r', track_order=True)
        self.cube_utils = cu.CubeUtils("../../config/SDSS_Bands",
                                       "../../config/ccd_gain.tsv",
                                       "../../config/ccd_dark_variance.tsv")
        self.reader = h5r.SDSSCubeReader(self.h5_file, self.cube_utils)
        self.resolution = 0

    def test_get_spectral_cube(self):
        data = self.reader.get_spectral_cube_for_res(0)
        assert data.shape[1] == 1

    def test_write_VO_table(self):
        self.output_path = "output.xml"
        self.reader.get_spectral_cube_for_res(0)
        self.reader.write_VOTable(self.output_path)
        self.send_samp("table.load.votable")
        assert True

    def test_write_FITS(self):
        self.output_path = "output.fits"
        self.reader.get_spectral_cube_for_res(self.resolution)
        self.reader.write_FITS(self.output_path)
        self.send_samp("table.load.fits")
        assert True

    def test_write_FITS_zoomed(self):
        self.resolution = 3
        self.test_write_FITS()

    def send_samp(self, message_type):
        client = SAMPIntegratedClient()
        client.connect()
        params = {"url": urljoin('file:', os.path.abspath(self.output_path)), "name": "SDSS Cube"}
        print(params["url"])
        message = {"samp.mtype": message_type, "samp.params": params}
        client.notify_all(message)
