import os
from urllib.parse import urljoin

from astropy.samp import SAMPIntegratedClient

from hisscube import FITSReader as FITS
from hisscube import Photometry as cu


class TestFITSCubeReader:

    def setup_method(self, test_method):
        spectra_path = "../../data/raw/galaxy_small/spectra"
        image_path = "../../data/raw/galaxy_small/images"
        self.reader = FITS.FITSReader(spectra_path, image_path, image_regex="frame-*-004136-*-0129.fits")
        self.resolution = 0

    def test_write_FITS(self):
        self.output_path = "../../results/output.fits"
        data = self.reader.get_spectral_cube_from_orig_for_res(self.resolution)
        self.reader.write_FITS(self.output_path)
        assert data.shape[1] == 1
        # self.send_samp("table.load.fits")

    def send_samp(self, message_type):
        client = SAMPIntegratedClient()
        client.connect()
        params = {"url": urljoin('file:', os.path.abspath(self.output_path)), "name": "SDSS Cube"}
        print(params["url"])
        message = {"samp.mtype": message_type, "samp.params": params}
        client.notify_all(message)
