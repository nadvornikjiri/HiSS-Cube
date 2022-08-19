import os
import subprocess
import warnings

import h5py

from hisscube.MLProcessor import MLProcessor
from hisscube.utils.io import SerialH5Connector

H5_DUMP_CMD_PATH = "../../ext_lib/hdf5-1.12.0/hdf5/bin/h5dump"
H5_DIFF_CMD_PATH = "../../ext_lib/hdf5-1.12.0/hdf5/bin/h5diff"


class TestHiSSCube:
    def test_serial_metadata(self):
        H5PATH = "../../results/SDSS_cube.h5"
        H5_DUMP_PATH = "../../results/SDSS_cube_dump.txt"
        H5_TESTDUMP_PATH = "../../results/SDSS_cube_test_dump.txt"
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        test_images = "../../data/raw/galaxy_small/images"
        image_pattern = "frame-*-004136-*-0129.fits"
        test_spectra = "../../data/raw/galaxy_small/spectra"
        spectra_pattern = "*.fits"
        self.writer = SerialH5Connector(h5_path=H5PATH)
        self.writer.CREATE_REFERENCES = True
        self.writer.CREATE_DENSE_CUBE = True
        self.writer.open_h5_file(truncate_file=True)
        self.writer.ingest(test_images, test_spectra, image_pattern, spectra_pattern)
        self.writer.close_h5_file()
        self.h5_file = h5py.File(H5PATH, 'r+', track_order=True, libver="latest")
        writer = MLProcessor(self.h5_file)
        writer.create_3d_cube()
        writer.close_h5_file()
        subprocess.call("%s -pBH %s > %s" % (H5_DUMP_CMD_PATH, H5PATH, H5_DUMP_PATH), shell=True,
                        executable='/bin/bash')
        diff_output = subprocess.run(
            "diff <(tail -n +2 %s) <(tail -n +2 %s)" % (H5_TESTDUMP_PATH, H5_DUMP_PATH), shell=True,
            executable='/bin/bash', capture_output=True, text=True).stdout  # check diff from second line

        assert (diff_output == "")

    def test_parallel_metadata(self):
        INPUT_PATH = "../../data/raw/galaxy_small/"
        H5_DUMP_PATH = "../../results/SDSS_cube_c_par_dump.txt"
        H5_TESTDUMP_PATH = "../../results/SDSS_cube_c_par_test_dump.txt"
        OUTPUT_PATH = "../../results/SDSS_cube_c_par.h5"
        PYTHON_PATH = "../../venv_par/bin/python"
        output = os.popen(
            "mpiexec -n 8 %s ../../hisscube.py --truncate %s %s ingest" % (PYTHON_PATH, INPUT_PATH, OUTPUT_PATH)).read()
        print(output)
        subprocess.call("%s -pBH %s > %s" % (H5_DUMP_CMD_PATH, OUTPUT_PATH, H5_DUMP_PATH), shell=True,
                        executable='/bin/bash')
        diff_output = subprocess.run(
            "diff <(tail -n +2 %s) <(tail -n +2 %s)" % (H5_TESTDUMP_PATH, H5_DUMP_PATH), shell=True,
            executable='/bin/bash', capture_output=True, text=True).stdout  # check diff from second line

        assert (diff_output == "")

    def test_serial_whole(self):
        H5PATH = "../../results/SDSS_cube.h5"
        H5_TESTPATH = "../../results/SDSS_cube_test.h5"
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        test_images = "../../data/raw/galaxy_small/images"
        image_pattern = "frame-*-004136-*-0129.fits"
        test_spectra = "../../data/raw/galaxy_small/spectra"
        spectra_pattern = "*.fits"
        self.writer = SerialH5Connector(h5_path=H5PATH)
        self.writer.CREATE_REFERENCES = True
        self.writer.CREATE_DENSE_CUBE = True
        self.writer.open_h5_file(truncate_file=True)
        self.writer.ingest(test_images, test_spectra, image_pattern, spectra_pattern)
        self.writer.close_h5_file()
        self.h5_file = h5py.File(H5PATH, 'r+', track_order=True, libver="latest")
        writer = MLProcessor(self.h5_file)
        writer.create_3d_cube()
        writer.close_h5_file()

        diff_output = subprocess.run("%s %s %s" % (H5_DIFF_CMD_PATH, H5PATH, H5_TESTPATH), shell=True,
                                     executable='/bin/bash', capture_output=True, text=True).stdout
        assert (diff_output == "")

    def test_parallel_whole(self):
        INPUT_PATH = "../../data/raw/galaxy_small/"
        OUTPUT_PATH = "../../results/SDSS_cube_c_par.h5"
        H5_TESTPATH = "../../results/SDSS_cube_c_par_test.h5"
        PYTHON_PATH = "../../venv_par/bin/python"
        output = os.popen(
            "mpiexec -n 8 %s ../../hisscube.py --truncate %s %s ingest" % (
                PYTHON_PATH, INPUT_PATH, OUTPUT_PATH)).read()
        print(output)
        diff_output = subprocess.run("%s %s %s" % (H5_DIFF_CMD_PATH, OUTPUT_PATH, H5_TESTPATH), shell=True,
                                     executable='/bin/bash', capture_output=True, text=True).stdout
        assert (diff_output == "")
