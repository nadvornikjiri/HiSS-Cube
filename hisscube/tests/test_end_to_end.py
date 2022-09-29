import os
import subprocess
import unittest
import warnings
from unittest.mock import Mock
import pytest

from hisscube.tests.test_serial_builders import get_test_director
from hisscube.utils.config import Config

H5_DUMP_CMD_PATH = "../../ext_lib/hdf5-1.12.0/hdf5/bin/h5dump"
H5_DIFF_CMD_PATH = "../../ext_lib/hdf5-1.12.0/hdf5/bin/h5diff"


def get_dump_strip_cmd(path):
    return "sed '/OFFSET/d' %s |sed '/HDF5/d'" % path


class TestHiSSCube(unittest.TestCase):
    def test_serial_metadata(self):
        h5_dump_path = "../../results/SDSS_cube_dump.txt"
        h5_testdump_path = "../../results/SDSS_cube_test_dump.txt"
        h5_path = self.construct_serial()
        diff_output = self.diff_dump(h5_dump_path, h5_path, h5_testdump_path)

        assert (diff_output == "")

    # @pytest.mark.skip(reason="Long run")
    def test_parallel_metadata(self):
        h5_dump_path = "../../results/SDSS_cube_c_par_dump.txt"
        h5_testdump_path = "../../results/SDSS_cube_c_par_test_dump.txt"
        h5_path = self.construct_parallel()

        diff_output = self.diff_dump(h5_dump_path, h5_path, h5_testdump_path)
        assert (diff_output == "")
        return h5_dump_path, h5_testdump_path, h5_path

    @pytest.mark.skip(reason="H5 Files not versioned with git.")
    def test_serial_whole(self):
        h5_test_path = "../../results/SDSS_cube_test.h5"
        h5_path = self.construct_serial()
        diff_output = self.h5diff_whole(h5_path, h5_test_path)
        assert (diff_output == "")

    @pytest.mark.skip(reason="H5 Files not versioned with git. Long run")
    def test_parallel_whole(self):
        h5_test_path = "../../results/SDSS_cube_c_par_test.h5"
        h5_path = self.construct_parallel()
        diff_output = self.h5diff_whole(h5_path, h5_test_path)
        assert (diff_output == "")

    @staticmethod
    def construct_serial():
        h5_path = "../../results/SDSS_cube.h5"
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        test_images = "../../data/raw/galaxy_small/images"
        image_pattern = "frame-*-004136-*-0129.fits"
        test_spectra = "../../data/raw/galaxy_small/spectra"
        spectra_pattern = "*.fits"
        args = Mock()
        args.command = "create"
        args.output_path = h5_path
        config = Config()
        config.METADATA_STRATEGY = "TREE"
        dependency_provider, director = get_test_director(args, test_images, test_spectra, image_pattern,
                                                          spectra_pattern, config=config)
        director.construct()
        return h5_path

    @staticmethod
    def construct_parallel():
        input_path = "../../data/raw/galaxy_small/"
        h5_path = "../../results/SDSS_cube_c_par.h5"
        python_path = "../../venv_par/bin/python"
        cmd = "mpiexec -n 8 %s ../../hisscube.py %s %s create" % (python_path, input_path, h5_path)
        print("Running command: %s" % cmd)
        output = os.popen(cmd).read()
        return h5_path

    @staticmethod
    def diff_dump(h5_dump_path, h5_path, h5_testdump_path):
        subprocess.call("%s -pBH %s > %s" % (H5_DUMP_CMD_PATH, h5_path, h5_dump_path), shell=True,
                        executable='/bin/bash')
        diff_cmd = "diff <(%s) <(%s)" % (get_dump_strip_cmd(h5_testdump_path), get_dump_strip_cmd(h5_dump_path))
        print(diff_cmd)
        diff_output = subprocess.run(diff_cmd, shell=True,
                                     executable='/bin/bash', capture_output=True,
                                     text=True).stdout  # check diff from second line
        return diff_output

    @staticmethod
    def h5diff_whole(h5_path, h5_test_path):
        diff_output = subprocess.run("%s %s %s" % (H5_DIFF_CMD_PATH, h5_path, h5_test_path), shell=True,
                                     executable='/bin/bash', capture_output=True, text=True).stdout
        return diff_output
