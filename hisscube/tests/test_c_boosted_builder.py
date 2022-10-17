from __future__ import print_function

import unittest

import h5py

from hisscube.dependency_injector import HiSSCubeProvider
from hisscube.utils.config import Config
from hisscube.utils.io import truncate

try:
    from reprlib import repr
except ImportError:
    pass

FITS_IMAGE_PATH = "../../data/raw/galaxy_small/images"
FITS_SPECTRA_PATH = "../../data/raw/galaxy_small/spectra"
H5_PATH = "../../results/SDSS_cube_c_par.h5"


class TestCBoostedBuilder(unittest.TestCase):

    def setup_method(self, test_method):
        truncate(H5_PATH)
        config = Config()
        config.METADATA_STRATEGY = "TREE"
        config.IMG_SPAT_INDEX_ORDER = 9
        self.dependency_provider = HiSSCubeProvider(H5_PATH, image_path=FITS_IMAGE_PATH,
                                                    spectra_path=FITS_SPECTRA_PATH, config=config)
        self.dependency_provider.config.MPIO = False
        fits_cache_builder = self.dependency_provider.serial_builders.metadata_cache_builder
        fits_cache_builder.build()

    def test_build(self):
        c_builder = self.dependency_provider.serial_builders.c_boosted_metadata_builder
        c_builder.build()
        h5_file = h5py.File(H5_PATH, libver="latest")
        test_ds = h5_file[
            "/semi_sparse_cube/5/22/90/362/1450/5802/23208/92833/371334/4604806627.9/6166/(1024, 744)/frame-r-004899-2-0260.fits"]
        orig_res_link = test_ds.attrs["orig_res_link"]
        orig_res_ds = h5_file[
            "/semi_sparse_cube/5/22/90/362/1450/5802/23208/92833/371334/4604806627.9/6166/(2048, 1489)/frame-r-004899-2-0260.fits"]
        orig_res_ds_name = orig_res_ds.name.split('/')[-1]
        test_ds_name = h5_file[orig_res_link].name.split('/')[-1]
        assert (orig_res_ds_name == test_ds_name)
