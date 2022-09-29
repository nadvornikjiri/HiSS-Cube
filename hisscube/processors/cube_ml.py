import h5py

import numpy as np

from hisscube.processors.metadata_strategy_cube_ml import MLProcessorStrategy
from hisscube.processors.metadata_strategy_spectrum import get_spectrum_time

from hisscube.utils.astrometry import get_cutout_pixel_coords
from hisscube.utils.io_strategy import get_orig_header
from hisscube.utils.logging import HiSSCubeLogger
from hisscube.utils.nexus import add_nexus_navigation_metadata, set_nx_data, set_nx_interpretation, set_nx_signal

class MLProcessor:
    def __init__(self, metadata_strategy: MLProcessorStrategy):
        self.metadata_strategy = metadata_strategy

    def create_3d_cube(self, h5_connector):
        self.metadata_strategy.create_3d_cube(h5_connector)

    def get_spectrum_3d_cube(self, h5_connector, zoom):
        return self.metadata_strategy.get_spectrum_3d_cube(h5_connector, zoom)

    def get_target_count(self, h5_connector):
        return self.metadata_strategy.get_target_count(h5_connector)
