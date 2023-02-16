import unittest

from hisscube.utils.config import Config
from hisscube.utils.io import truncate, ParallelH5Writer
from hisscube.utils.io_strategy import ParallelDatasetIOStrategy

H5_PATH = "../../results/SDSS_cube.h5"
INPUT_PATH = "../../data/raw"


def get_default_config():
    config = Config()
    config.MPIO = True
    config.C_BOOSTER = False
    config.USE_SUBFILING = True
    config.METADATA_STRATEGY = "DATASET"
    return config


class TestSerialBuilder(unittest.TestCase):
    def setup_method(self, test_method):
        self.config = get_default_config()

    def test_open(self):
        io_strategy = ParallelDatasetIOStrategy()
        writer = ParallelH5Writer(H5_PATH, self.config, io_strategy)
        writer.open_h5_file(truncate_file=True)
        writer.close_h5_file()

