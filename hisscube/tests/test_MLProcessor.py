import h5py

from hisscube.Photometry import Photometry
from hisscube.MLProcessor import MLProcessor

H5PATH = "../../data/processed/SDSS_cube.h5"


class TestMLProcessor:
    def setup_method(self, test_method):
        self.h5_file = h5py.File(H5PATH, 'r+', track_order=True, libver="latest")

    def teardown_method(self, test_method):
        self.h5_file.close()

    def test_create_3d_cube(self):
        writer = MLProcessor(self.h5_file)
        writer.create_3d_cube()
        assert True

    def test_count_spatial_groups_with_depth(self):
        processor = MLProcessor(self.h5_file)
        target_cnt = processor.count_spatial_groups_with_depth(
            processor.f[processor.config.get("Handler", "ORIG_CUBE_NAME")],
            processor.config.getint("Handler", "SPEC_SPAT_INDEX_ORDER"))
        assert (target_cnt == 3)
