import fitsio
from scripts import SDSSCubeWriter as h5u
from scripts import cubeUtils as cu
import h5py
import pytest
import glob
import numpy as np
from pathlib import Path

H5PATH = "../../data/SDSS_cube.h5"

@pytest.fixture(scope="session", autouse=False)
def truncate_test_file(request):
    h5path = H5PATH
    f = h5py.File(h5path, 'w')  # create + truncate file
    f.close()


class TestH5Writer:

    def setup_method(self, test_method):
        self.h5_file = h5py.File(H5PATH, 'r+', track_order=True)
        self.cube_utils = cu.CubeUtils("../../config/SDSS_Bands",
                                       "../../config/ccd_gain.tsv",
                                       "../../config/ccd_dark_variance.tsv")

    def teardown_method(self, test_method):
        self.h5_file.close()

    @pytest.mark.usefixtures("truncate_test_file")
    def test_add_image(self):
        test_path = "../../data/images/301/4797/1/frame-g-004797-1-0019.fits.bz2"

        writer = h5u.SDSSCubeWriter(self.h5_file, self.cube_utils)
        h5_datasets = writer.ingest_image(test_path)
        assert len(h5_datasets) == 4

    @pytest.mark.usefixtures("truncate_test_file")
    def test_add_spectrum(self):
        test_path = "../../data/spectra/spec-4500-55543-0331.fits"

        writer = h5u.SDSSCubeWriter(self.h5_file, self.cube_utils)
        h5_datasets = writer.ingest_spectrum(test_path)
        assert len(h5_datasets) == 5

    @pytest.mark.usefixtures("truncate_test_file")
    def test_add_image_multiple(self):
        test_images_small = "../../data/images/301/2820/3/frame-*-002820-3-0122.fits.bz2"
        writer = h5u.SDSSCubeWriter(self.h5_file, self.cube_utils)
        for image in glob.glob(test_images_small):
            print("writing: %s" % image)
            writer.ingest_image(image)

    def test_add_spec_refs(self):
        test_spectrum = "../../data/spectra/3126/spec-7330-56684-0434.fits"
        writer = h5u.SDSSCubeWriter(self.h5_file, self.cube_utils)
        spec_datasets = writer.ingest_spectrum(test_spectrum)
        spec_datasets = writer.add_spec_refs(spec_datasets)
        assert len(spec_datasets[0].attrs["image_cutouts"]) == 5
        assert type(spec_datasets[0].attrs["image_cutouts"][0]) is h5py.RegionReference

    def test_find_images_overlapping_spectrum(self):
        test_spectrum = "../../data/spectra/3126/spec-7330-56684-0434.fits"
        writer = h5u.SDSSCubeWriter(self.h5_file, self.cube_utils)
        with fitsio.FITS(test_spectrum) as hdul:
            writer.metadata = hdul[0].read_header()
        writer.find_images_overlapping_spectrum()

    def test_crop_cutout_to_image(self):
        writer = h5u.SDSSCubeWriter(self.h5_file, self.cube_utils)
        top_left, top_right, bot_left, bot_right = [-32, -32], [-32, 32], [32, -32], [32, 32]
        e_top_left, e_top_right, e_bot_left, e_bot_right = [0, 0], [0, 32], [32, 0], [32, 32]
        h5u.SDSSCubeWriter.crop_cutout_to_image(top_left, top_right, bot_left, bot_right, np.array([2048, 1489]))
        assert (top_left == e_top_left)
        assert (top_right == e_top_right)
        assert (bot_left == e_bot_left)
        assert (top_right == e_top_right)

        top_left, top_right, bot_left, bot_right = [1488, 2047], [1488, 2111], [1552, 2047], [1552, 2111]
        e_top_left, e_top_right, e_bot_left, e_bot_right = [1488, 2047], [1488, 2047], [1488, 2047], [1488, 2047]
        h5u.SDSSCubeWriter.crop_cutout_to_image(top_left, top_right, bot_left, bot_right, np.array([2048, 1489]))
        assert (top_left == e_top_left)
        assert (top_right == e_top_right)
        assert (bot_left == e_bot_left)
        assert (top_right == e_top_right)

        top_left, top_right, bot_left, bot_right = [500, 500], [500, 500], [500, 500], [500, 500],
        e_top_left, e_top_right, e_bot_left, e_bot_right = [500, 500], [500, 500], [500, 500], [500, 500],
        h5u.SDSSCubeWriter.crop_cutout_to_image(top_left, top_right, bot_left, bot_right, np.array([2048, 1489]))
        assert (top_left == e_top_left)
        assert (top_right == e_top_right)
        assert (bot_left == e_bot_left)
        assert (top_right == e_top_right)

    def test_add_spectra_multiple(self):
        spectra_folder = "../../data/spectra/3126"
        test_spectra_small = "*.fits"
        writer = h5u.SDSSCubeWriter(self.h5_file, self.cube_utils)
        for spectrum in Path(spectra_folder).rglob(test_spectra_small):
            print("writing: %s" % spectrum)
            datasets = writer.ingest_spectrum(spectrum)
            assert (len(datasets) > 1)

