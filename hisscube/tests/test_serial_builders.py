import os
import unittest
import warnings
from pathlib import Path
from unittest.mock import Mock

import h5py
import pytest
from tqdm.auto import tqdm

from hisscube.dependency_injector import HiSSCubeProvider
from hisscube.director import HiSSCubeConstructionDirector
from hisscube.processors.data import float_compress
from hisscube.utils.astrometry import is_cutout_whole
from hisscube.utils.config import Config
from hisscube.utils.io import truncate, H5Connector
from hisscube.utils.logging import wrap_tqdm

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import numpy as np

H5_PATH = "../../results/SDSS_cube.h5"
INPUT_PATH = "../../data/raw"


def get_default_config():
    config = Config()
    config.MPIO = False
    config.C_BOOSTER = False
    config.METADATA_STRATEGY = "TREE"
    return config


def get_test_director(args, test_images, test_spectra, image_pattern, spectra_pattern, config=None):
    if not config:
        config = get_default_config()
    dependency_provider = HiSSCubeProvider(H5_PATH, image_path=test_images, spectra_path=test_spectra,
                                           image_pattern=image_pattern, spectra_pattern=spectra_pattern, config=config)
    dependency_provider.config.MPIO = False
    director = HiSSCubeConstructionDirector(args, dependency_provider.config, dependency_provider.serial_builders,
                                            dependency_provider.parallel_builders)
    return dependency_provider, director


class TestSerialBuilder(unittest.TestCase):

    def setup_method(self, test_method):
        self.config = get_default_config()
        truncate(H5_PATH, self.config)
        self.dependency_provider = HiSSCubeProvider(H5_PATH, input_path=INPUT_PATH, config=self.config)

        warnings.filterwarnings("ignore", category=DeprecationWarning)

    def test_build_image_single(self):
        test_path = "../../data/raw/galaxy_small/images/301/4136/3/frame-r-004136-3-0129.fits"
        builder = self.dependency_provider.serial_builders.single_image_builder
        builder.image_path = test_path
        h5_datasets = builder.build()
        assert len(h5_datasets) == self.dependency_provider.config.IMG_ZOOM_CNT

    def test_build_image_single_dataset_strategy(self):
        self.config.METADATA_STRATEGY = "DATASET"
        self.dependency_provider = HiSSCubeProvider(H5_PATH, input_path=INPUT_PATH, config=self.config)
        self.test_build_image_single()

    def test_build_spectrum_single(self):
        test_path = "../../data/raw/problematic/spectra/spec-5290-55862-0984.fits"
        builder = self.dependency_provider.serial_builders.single_spectrum_builder
        builder.spectrum_path = test_path
        h5_datasets = builder.build()
        assert len(h5_datasets) == self.dependency_provider.config.SPEC_ZOOM_CNT

    def test_build_spetrum_single_dataset_strategy(self):
        self.config.METADATA_STRATEGY = "DATASET"
        self.dependency_provider = HiSSCubeProvider(H5_PATH, input_path=INPUT_PATH, config=self.config)
        self.test_build_spectrum_single()

    @staticmethod
    def get_image_test(image_pattern, test_images, config=None):
        test_spectra = ""
        spectra_pattern = ""
        args = Mock()
        args.command = "update"
        args.fits_metadata_cache = True
        args.metadata = True
        args.data = True
        args.link = False
        args.visualization_cube = False
        args.ml_cube = False
        args.sfr = False
        args.output_path = H5_PATH
        dependency_provider, director = get_test_director(args, test_images, test_spectra, image_pattern,
                                                          spectra_pattern, config)
        return dependency_provider, director

    def test_add_image_multiple(self):
        test_images = "../../data/raw/galaxy_small/images"
        image_pattern = "frame-*-004136-*-0129.fits"
        dependency_provider, director = self.get_image_test(image_pattern, test_images)
        director.construct()

    def test_add_image_multiple_dataset_strategy(self):
        self.config.METADATA_STRATEGY = "DATASET"
        test_images = "../../data/raw/galaxy_small/images"
        image_pattern = "frame-*-004136-*-0129.fits"
        dependency_provider, director = self.get_image_test(image_pattern, test_images, self.config)

    def test_add_image_multiple_same_run(self):
        test_images = "../../data/raw/problematic/images_same_run"
        image_pattern = "*.fits"
        dependency_provider, director = self.get_image_test(image_pattern, test_images)

        dependency_provider.config.IMG_SPAT_INDEX_ORDER = 8
        with self.assertRaises(AssertionError):
            director.construct()

    def test_add_spec_refs(self):
        dependency_provider = self.run_link_spectra()
        with dependency_provider.h5_serial_writer as h5_connector:
            spec_datasets = self._get_test_spec_ds(h5_connector.file)
            self._assert_image_cutout_sizes(h5_connector, spec_datasets)

    def run_link_spectra(self, config=None):
        image_pattern = "frame-r-004136-3-0129.fits"
        spectra_pattern = "spec-0412-51871-0308.fits"
        return self._insert_links(image_pattern, spectra_pattern, config)

    @staticmethod
    def _get_test_spec_ds(h5_file):
        spec_datasets = []
        test_grp = h5_file[
            "/semi_sparse_cube/5/22/90/362/1450/5802/23208/92835/371341/1485364/5941459/23765838/95063352/380253411/1521013644/6084054576/4481683956.2"]
        for res_grp_name in test_grp:
            res_grp = test_grp[res_grp_name]
            spec_ds_name = list(res_grp)[0]
            spec_ds = res_grp[spec_ds_name]
            spec_datasets.append(spec_ds)
        return spec_datasets

    def _assert_image_cutout_sizes(self, h5_connector, spec_datasets):
        assert bool(spec_datasets[0].parent.parent.parent["image_cutouts_0"][0])
        for spec_dataset in spec_datasets:
            for zoom in range(self.dependency_provider.config.SPEC_ZOOM_CNT):
                cutout_list = spec_dataset.parent.parent.parent["image_cutouts_%d" % zoom]
                self.assert_cutouts(h5_connector, cutout_list, 0, 1)

    def assert_cutouts(self, h5_connector: H5Connector, cutout_list, x_idx, y_idx):
        assert cutout_list[0]
        for cutout in cutout_list:
            if not cutout:
                break
            cutout = h5_connector.dereference_region_ref(cutout)
            cutout_shape = cutout.shape
            assert (0 <= cutout_shape[x_idx] <= 64)
            assert (0 <= cutout_shape[y_idx] <= 64)

    def test_add_spec_refs_dataset_strategy(self):
        self.config.METADATA_STRATEGY = "DATASET"
        dependency_provider = self.run_link_spectra(self.config)
        with dependency_provider.h5_serial_writer as h5_connector:
            cutout_list = self._get_test_cutout_ds(h5_connector.file)
            self.assert_cutouts(h5_connector, cutout_list, 0, 1)

    @staticmethod
    def _get_test_cutout_ds(h5_file):
        return h5_file["semi_sparse_cube/0/spectra/image_cutouts_data"][0]

    @staticmethod
    def _insert_links(image_pattern, spectra_pattern, config=None):
        test_images = "../../data/raw/galaxy_small/images"
        test_spectra = "../../data/raw/galaxy_tiny/spectra"
        args = Mock()
        args.truncate = True
        args.command = "update"
        args.fits_metadata_cache = True
        args.metadata = True
        args.data = True
        args.link_images_spectra = True
        args.visualization_cube = False
        args.ml_cube = False
        args.sfr = False
        args.output_path = H5_PATH
        dependency_provider, director = get_test_director(args, test_images, test_spectra, image_pattern,
                                                          spectra_pattern, config=config)
        director.construct()
        return dependency_provider

    def test_is_cutout_whole(self):
        test1 = [[[735, 1849],
                  [799, 1849]],
                 [[735, 1913],
                  [799, 1913]]]
        test2 = [[[735, 1849],
                  [799, 1849]],
                 [[735, 1913],
                  [799, 1913]]]
        test3 = [[[-1, 1849],
                  [63, 1849]],
                 [[-1, 1913],
                  [-1, 1913]]]
        test4 = [[[735, 64],
                  [799, 64]],
                 [[735, 128],
                  [799, 128]]]
        tests = [test1, test2, test3, test4]
        expected = [False, False, False, True]
        results = []
        for test in tests:
            results.append(is_cutout_whole(test, np.zeros((1849, 2048, 2))))
        assert (results == expected)

    def test_rebin(self):
        spectra_folder = "../../data/raw/galaxy_small/spectra"
        spectra_pattern = "*.fits"
        photometry = self.dependency_provider.photometry
        config = self.dependency_provider.config
        spectra_paths = list(Path(spectra_folder).rglob(spectra_pattern))
        iterator = wrap_tqdm(spectra_paths, self.config.MPIO, self.__class__.__name__, self.config)
        for spec_path in iterator:
            metadata, data = photometry.get_multiple_resolution_spectrum(
                spec_path,
                config.SPEC_ZOOM_CNT,
                apply_rebin=config.APPLY_REBIN,
                rebin_min=config.REBIN_MIN,
                rebin_max=config.REBIN_MAX,
                rebin_samples=config.REBIN_SAMPLES,
                apply_transmission=config.APPLY_TRANSMISSION_CURVE)
            assert (data[0]['flux_mean'].shape[0] == 4620)

    @pytest.mark.skip(reason="Long run")
    def test_float_compress(self):
        test_path = "../../data/raw/images/301/2820/3/frame-g-002820-3-0122.fits"
        self.dependency_provider.config.FLOAT_COMPRESS = True
        metadata_processor = self.dependency_provider.processors.metadata_processor
        data_processor = self.dependency_provider.processors.data_processor
        photometry = self.dependency_provider.photometry
        config = self.dependency_provider.config
        metadata_processor.spectrum_metadata, data_processor.data = photometry.get_multiple_resolution_image(
            test_path,
            config.IMG_ZOOM_CNT)
        metadata_processor.file_name = os.path.basename(test_path)
        for img in data_processor.data:  # parsing 2D resolution
            img_data = np.dstack((img["flux_mean"], img["flux_sigma"]))
            float_compressed_data = float_compress(img_data)
            # check that we truncated the mantissa correctly - 13 last digits of mantissa should be 0
            for compressed_number, orig_number in np.nditer((float_compressed_data, img_data)):
                if orig_number != 0:
                    if bin(compressed_number.view("i"))[-13:] != "0000000000000":
                        print("Failed at elem %f, %d: " % (compressed_number, orig_number))
                        assert False
                    if abs((compressed_number / orig_number) - 1) >= 0.01:
                        print("Failed at elements %f, %d, error %f" % (
                            compressed_number, orig_number, np.abs((compressed_number / orig_number) - 1)))
                        assert False
        assert True
