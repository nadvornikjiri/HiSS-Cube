import unittest
import warnings
from unittest.mock import Mock

from hisscube.dependency_injector import HiSSCubeProvider
from hisscube.director import HiSSCubeConstructionDirector
from hisscube.processors.sfr import SFRProcessor
from hisscube.utils.config import Config

H5PATH = "../../results/SDSS_cube.h5"
GAL_INFO_PATH = "../../data/SFR/gal_info_dr7_v5_2.fit"
GAL_SFR_PATH = "../../data/SFR/gal_fibsfr_dr7_v5_2.fits"


class TestExport(unittest.TestCase):

    metadata_strategy = "DATASET"

    def setup_class(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        test_images = "../../data/raw/galaxy_small/images"
        image_pattern = "frame-*-004136-*-0129.fits"
        test_spectra = "../../data/raw/galaxy_small/spectra"
        spectra_pattern = "*.fits"
        args = Mock()
        args.command = "update"
        args.truncate = True
        args.fits_metadata_cache = True
        args.metadata = True
        args.data = False
        args.data_image = False
        args.data_spectrum  = False
        args.link = False
        args.visualization_cube = False
        args.ml_cube = False
        args.sfr = False
        args.output_path = H5PATH
        self.config = Config()
        self.config.MPIO = False
        self.config.METADATA_STRATEGY = self.metadata_strategy
        self.dependency_provider = HiSSCubeProvider(H5PATH, image_path=test_images, spectra_path=test_spectra,
                                                    image_pattern=image_pattern, spectra_pattern=spectra_pattern,
                                                    config=self.config)
        director = HiSSCubeConstructionDirector(args, self.dependency_provider.config,
                                                self.dependency_provider.serial_builders,
                                                self.dependency_provider.parallel_builders)
        director.construct()

    def test_write_sfr(self):
        processor = SFRProcessor()
        with self.dependency_provider.h5_pandas_writer as h5_connector:
            sfr_df = processor.write_sfr(h5_connector, GAL_INFO_PATH, GAL_SFR_PATH)
        with self.dependency_provider.h5_serial_reader as h5_connector:
            spec_headers_df = processor.get_spectrum_metadata(h5_connector)
        with self.dependency_provider.h5_pandas_writer as h5_connector:
            merged_df = processor.write_spec_metadata_with_sfr(h5_connector, spec_headers_df, sfr_df)
        assert merged_df.shape == (11, 170)
        assert merged_df["MEDIAN"].isna().sum() == 4
        with self.dependency_provider.h5_pandas_writer as h5_connector:
            test_df = h5_connector.file["fits_spectra_metadata_star_formation_rates"]
            assert test_df.shape == (11, 170)
            assert test_df["MEDIAN"].isna().sum() == 4
