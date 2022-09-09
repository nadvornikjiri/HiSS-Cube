import os
import unittest
import warnings
from pathlib import Path
from unittest.mock import Mock

import fitsio
import h5py
import pytest
from tqdm.auto import tqdm

from hisscube.builders import SingleImageBuilder, HiSSCubeConstructionDirector
from hisscube.dependency_injector import HiSSCubeProvider
from hisscube.processors.data import float_compress
from hisscube.utils.io import SerialH5Writer, truncate
from hisscube.utils.astrometry import is_cutout_whole

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import numpy as np

H5_PATH = "../../results/SDSS_cube.h5"
INPUT_PATH = "../../data/raw"


def get_test_director(args, test_images, test_spectra, image_pattern, spectra_pattern):
    dependency_provider = HiSSCubeProvider(H5_PATH, image_path=test_images, spectra_path=test_spectra,
                                           image_pattern=image_pattern, spectra_pattern=spectra_pattern)
    dependency_provider.config.MPIO = False
    director = HiSSCubeConstructionDirector(args, dependency_provider.config, dependency_provider.serial_builders,
                                            dependency_provider.parallel_builders)
    return dependency_provider, director


class TestSerialBuilder(unittest.TestCase):

    def setup_method(self, test_method):
        truncate(H5_PATH)
        self.dependency_provider = HiSSCubeProvider(H5_PATH, input_path=INPUT_PATH)
        self.dependency_provider.config.MPIO = False
        self.dependency_provider.config.C_BOOSTER = False
        warnings.filterwarnings("ignore", category=DeprecationWarning)

    def test_build_image_single(self):
        test_path = "../../data/raw/galaxy_small/images/frame-r-004136-3-0129.fits"
        builder = self.dependency_provider.serial_builders.single_image_builder
        builder.image_path = test_path
        h5_datasets = builder.build()
        assert len(h5_datasets) == self.dependency_provider.config.IMG_ZOOM_CNT

    def test_build_spetrum_single(self):
        test_path = "../../data/raw/problematic/spectra/spec-5290-55862-0984.fits"
        builder = self.dependency_provider.serial_builders.single_spectrum_builder
        builder.spectrum_path = test_path
        h5_datasets = builder.build()
        assert len(h5_datasets) == self.dependency_provider.config.SPEC_ZOOM_CNT

    def test_add_image_multiple_same_run(self):
        test_images = "../../data/raw/problematic/images_same_run"
        image_pattern = "*.fits"
        test_spectra = ""
        spectra_pattern = ""
        args = Mock()
        args.command = "create"
        args.output_path = H5_PATH
        dependency_provider, director = get_test_director(args, test_images, test_spectra, image_pattern,
                                                          spectra_pattern)

        dependency_provider.config.IMG_SPAT_INDEX_ORDER = 8
        with self.assertRaises(RuntimeError):
            director.construct()

    def test_add_spec_refs(self):
        image_pattern = "frame-r-004136-3-0129.fits"
        spectra_pattern = "spec-0412-51871-0308.fits"
        self._insert_links(image_pattern, spectra_pattern)
        with h5py.File(H5_PATH) as h5_file:
            spec_datasets = self._get_test_spec_ds(h5_file)
            self._assert_image_cutout_sizes(h5_file, spec_datasets)

    @staticmethod
    def _insert_links(image_pattern, spectra_pattern):
        test_images = "../../data/raw/galaxy_small/images"
        test_spectra = "../../data/raw/galaxy_tiny/spectra"
        args = Mock()
        args.command = "update"
        args.fits_header_cache = True
        args.metadata = True
        args.data = True
        args.link_images_spectra = True
        args.output_path = H5_PATH
        dependency_provider, director = get_test_director(args, test_images, test_spectra, image_pattern,
                                                          spectra_pattern)
        director.construct()

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

    def _assert_image_cutout_sizes(self, h5_file, spec_datasets):
        assert bool(spec_datasets[0].parent.parent.parent["image_cutouts_0"][0])
        for spec_dataset in spec_datasets:
            for zoom in range(self.dependency_provider.config.SPEC_ZOOM_CNT):
                for cutout in spec_dataset.parent.parent.parent["image_cutouts_%d" % zoom]:
                    if not cutout:
                        break
                    cutout_shape = h5_file[cutout][cutout].shape
                    assert (0 <= cutout_shape[0] <= 64)
                    assert (0 <= cutout_shape[1] <= 64)
                    assert (cutout_shape[2] == 2)

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
        for spec_path in tqdm(spectra_paths, desc="Spectra completed: "):
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
        metadata_processor.metadata, data_processor.data = photometry.get_multiple_resolution_image(
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
