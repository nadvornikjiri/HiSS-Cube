import os
import timeit
import warnings
from urllib.parse import urljoin

import h5py
from astropy.samp import SAMPIntegratedClient

import pytest
from hisscube import VisualizationProcessor as h5r
from hisscube.utils.io import SerialH5Connector

H5PATH = "../../results/SDSS_cube.h5"


@pytest.mark.incremental
class TestH5Reader:

    def setup_class(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        test_images = "../../data/raw/galaxy_small/images"
        image_pattern = "frame-*-004136-*-0129.fits"
        test_spectra = "../../data/raw/galaxy_small/spectra"
        spectra_pattern = "*.fits"
        self.writer = SerialH5Connector(h5_path=H5PATH)
        self.writer.CREATE_REFERENCES = True
        self.writer.CREATE_DENSE_CUBE = True
        self.writer.open_h5_file(truncate_file=True)
        self.writer.ingest(test_images, test_spectra, image_pattern, spectra_pattern)
        self.writer.close_h5_file()

    def setup_method(self, test_method):
        self.h5_file = h5py.File(H5PATH, 'r', track_order=True, libver="latest")
        self.resolution = 0

    def teardown_method(self, test_method):
        self.h5_file.close()

    def test_get_spectral_cube_from_orig(self):
        self.reader = h5r.VisualizationProcessor(self.h5_file)
        data = self.reader.construct_spectral_cube_table(0)
        assert data.shape == (276100,)

    def test_get_spectral_cube(self):
        self.reader = h5r.VisualizationProcessor(self.h5_file)
        data = self.reader.read_spectral_cube_table(0)
        assert data.shape == (276100,)

    def test_write_VO_table(self):
        self.reader = h5r.VisualizationProcessor(self.h5_file)
        self.output_path = "../../results/output.xml"
        self.reader.construct_spectral_cube_table(0)
        self.reader.write_VOTable(self.output_path)
        # self.send_samp("table.load.votable")
        assert self.reader.spectral_cube.shape == (276100,)

    def test_write_FITS(self):
        self.reader = h5r.VisualizationProcessor(self.h5_file)
        self.output_path = "../../results/output.fits"
        self.reader.construct_spectral_cube_table(self.resolution)
        self.reader.write_FITS(self.output_path)
        # self.send_samp("table.load.fits")
        assert self.reader.spectral_cube.shape == (276100,)

    def test_write_FITS_zoomed(self):
        self.reader = h5r.VisualizationProcessor(self.h5_file)
        self.resolution = 3
        self.output_path = "../../results/output.fits"
        self.reader.construct_spectral_cube_table(self.resolution)
        self.reader.write_FITS(self.output_path)
        # self.send_samp("table.load.fits")
        assert self.reader.spectral_cube.shape == (9867,)

    def test_write_FITS_from_dense(self):
        self.output_path = "../../results/output.fits"
        start_time = timeit.default_timer()
        self.reader = h5r.VisualizationProcessor(self.h5_file)
        self.reader.read_spectral_cube_table(0)
        print(timeit.default_timer() - start_time)
        self.reader.write_FITS(self.output_path)
        # self.send_samp("table.load.fits")
        assert True

    def send_samp(self, message_type):
        client = SAMPIntegratedClient()
        client.connect()
        params = {"url": urljoin('file:', os.path.abspath(self.output_path)), "name": "SDSS Cube"}
        print(params["url"])
        message = {"samp.mtype": message_type, "samp.params": params}
        client.notify_all(message)
