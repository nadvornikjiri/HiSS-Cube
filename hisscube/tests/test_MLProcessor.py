import os
import warnings
from urllib.parse import urljoin

import h5py
from astropy.samp import SAMPIntegratedClient

from hisscube.Photometry import Photometry
from hisscube.MLProcessor import MLProcessor
from hisscube.VisualizationProcessor import VisualizationProcessor
from hisscube.Writer import Writer
import hashlib

H5PATH = "../../results/SDSS_cube.h5"


class TestMLProcessor:
    def setup_class(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        test_images = "../../data/raw/galaxy_small/images"
        image_pattern = "frame-*-004136-*-0129.fits"
        test_spectra = "../../data/raw/galaxy_small/spectra"
        spectra_pattern = "*.fits"
        self.writer = Writer(h5_path=H5PATH)
        self.writer.CREATE_REFERENCES = True
        self.writer.CREATE_DENSE_CUBE = True
        self.writer.open_h5_file_serial(truncate=True)
        self.writer.ingest(test_images, test_spectra, image_pattern, spectra_pattern)
        self.writer.close_h5_file()

    def setup_method(self, test_method):
        self.h5_file = h5py.File(H5PATH, 'r+', track_order=True, libver="latest")

    def teardown_method(self, test_method):
        self.h5_file.close()

    def test_create_3d_cube(self):
        writer = MLProcessor(self.h5_file)
        writer.create_3d_cube()
        # self.h5_file.close()
        # h5_file = h5py.File(H5PATH, 'r', track_order=True, libver="latest")
        # self.resolution = 0
        # reader = VisualizationProcessor(h5_file)
        # self.output_path = "../../results/output.fits"
        # reader.read_spectral_cube_table(0)
        # reader.write_FITS(self.output_path)
        # self.send_samp("table.load.fits")
        # h5_file.close()
        assert True

    def test_count_spatial_groups_with_depth(self):
        processor = MLProcessor(self.h5_file)
        target_cnt = processor.count_spatial_groups_with_depth(
            processor.f[processor.ORIG_CUBE_NAME],
            processor.config.getint("Handler", "SPEC_SPAT_INDEX_ORDER"))
        assert (target_cnt == 2)

    def send_samp(self, message_type):
        client = SAMPIntegratedClient()
        client.connect()
        params = {"url": urljoin('file:', os.path.abspath(self.output_path)), "name": "SDSS Cube"}
        print(params["url"])
        message = {"samp.mtype": message_type, "samp.params": params}
        client.notify_all(message)
