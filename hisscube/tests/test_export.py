import os
import timeit
import warnings
from unittest.mock import Mock
from urllib.parse import urljoin

import h5py
from astropy.samp import SAMPIntegratedClient

import pytest

from hisscube.builders import HiSSCubeConstructionDirector
from hisscube.dependency_injector import HiSSCubeProvider
from hisscube.processors.cube_ml import MLProcessor
from hisscube.processors.cube_visualization import VisualizationProcessor
from hisscube.utils.config import Config
from hisscube.utils.io import SerialH5Writer

H5PATH = "../../results/SDSS_cube.h5"


@pytest.mark.incremental
class TestExport:

    def setup_class(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        test_images = "../../data/raw/galaxy_small/images"
        image_pattern = "frame-*-004136-*-0129.fits"
        test_spectra = "../../data/raw/galaxy_small/spectra"
        spectra_pattern = "*.fits"
        args = Mock()
        args.command = "create"
        args.output_path = H5PATH
        self.config = Config()
        self.config.METADATA_STRATEGY = "TREE"
        self.dependency_provider = HiSSCubeProvider(H5PATH, image_path=test_images, spectra_path=test_spectra,
                                                    image_pattern=image_pattern, spectra_pattern=spectra_pattern,
                                                    config=self.config)
        self.dependency_provider.config.MPIO = False
        self.dependency_provider.config.INIT_ARRAY_SIZE = 10000
        director = HiSSCubeConstructionDirector(args, self.dependency_provider.config,
                                                self.dependency_provider.serial_builders,
                                                self.dependency_provider.parallel_builders)
        director.construct()

    def setup_method(self, test_method):
        self.resolution = 0

    def test_get_spectral_cube(self):
        with self.dependency_provider.h5_serial_reader as h5_connector:
            processor = VisualizationProcessor(self.config)
            processor.h5_connector = h5_connector
            data = processor.read_spectral_cube_table(0)
            assert data.shape == (276100,)

    def test_write_VO_table(self):
        with self.dependency_provider.h5_serial_reader as h5_connector:
            processor = VisualizationProcessor(self.config)
            processor.h5_connector = h5_connector
            self.output_path = "../../results/output.xml"
            processor.read_spectral_cube_table(0)
            processor.write_VOTable(self.output_path)
            # self.send_samp("table.load.votable")
            assert processor.spectral_cube.shape == (276100,)

    def test_write_fits(self):
        with self.dependency_provider.h5_serial_reader as h5_connector:
            processor = VisualizationProcessor(self.config)
            processor.h5_connector = h5_connector
            self.output_path = "../../results/output.fits"
            processor.read_spectral_cube_table(self.resolution)
            processor.write_FITS(self.output_path)
            # self.send_samp("table.load.fits")
            assert processor.spectral_cube.shape == (276100,)

    def test_write_fits_zoomed(self):
        with self.dependency_provider.h5_serial_reader as h5_connector:
            processor = VisualizationProcessor(self.config)
            processor.h5_connector = h5_connector

            self.output_path = "../../results/output.fits"
            processor.read_spectral_cube_table(zoom=3)
            processor.write_FITS(self.output_path)
            # self.send_samp("table.load.fits")
            assert processor.spectral_cube.shape == (9867,)

    def test_get_3d_cube(self):
        with self.dependency_provider.h5_serial_reader as h5_connector:
            processor = MLProcessor(self.config)
            processor.h5_connector = h5_connector
            cube_3d = processor.get_spectrum_3d_cube(zoom=2)
            assert cube_3d[0].shape == (2, 5, 16, 16)
            assert cube_3d[2].shape == (2, 1155)

    def send_samp(self, message_type):
        client = SAMPIntegratedClient()
        client.connect()
        params = {"url": urljoin('file:', os.path.abspath(self.output_path)), "name": "SDSS Cube"}
        print(params["url"])
        message = {"samp.mtype": message_type, "samp.params": params}
        client.notify_all(message)
