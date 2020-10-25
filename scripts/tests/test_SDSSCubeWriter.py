import os

import fitsio
from scripts import SDSSCubeWriter as h5u
from scripts import photometry as cu
import h5py
import pytest
import numpy as np
from pathlib import Path
import time

H5PATH = "../../SDSS_cube.h5"


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
    def test_add_image2(self):
        test_path = "../../data/images_medium_ds/frame-u-004948-3-0199.fits.bz2"

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
        # test_images = "../../data/images/301/2820/3/frame-*-002820-3-0122.fits.bz2"
        # test_images = "../../data/images_medium_ds"
        test_images = "../../data/galaxy_small/images"
        image_pattern = "*.fits*"
        writer = h5u.SDSSCubeWriter(self.h5_file, self.cube_utils)
        t0 = time.clock()
        for image in Path(test_images).rglob(image_pattern):
            print("Time: %5f writing: %s" % (time.clock() - t0, image))
            writer.ingest_image(image)

    def test_add_spec_refs(self):
        test_spectrum = "../../data/spectra/3126/spec-7330-56684-0434.fits"
        writer = h5u.SDSSCubeWriter(self.h5_file, self.cube_utils)
        spec_datasets = writer.ingest_spectrum(test_spectrum)
        spec_datasets = writer.add_image_refs_to_spectra(spec_datasets)
        for spec_dataset in spec_datasets:
            for cutout in spec_dataset.attrs["image_cutouts"]:
                cutout_shape = self.h5_file[cutout][cutout].shape
                assert (0 <= cutout_shape[0] <= 64)
                assert (0 <= cutout_shape[1] <= 64)
                assert (cutout_shape[2] == 2)

    def test_add_spec_refs2(self):
        test_spectrum = "../../data/spectra_medium_ds/spec-4238-55455-0037.fits"
        writer = h5u.SDSSCubeWriter(self.h5_file, self.cube_utils)
        spec_datasets = writer.ingest_spectrum(test_spectrum)
        spec_datasets = writer.add_image_refs_to_spectra(spec_datasets)
        for spec_dataset in spec_datasets:
            for cutout in spec_dataset.attrs["image_cutouts"]:
                cutout_shape = self.h5_file[cutout][cutout].shape
                assert (0 <= cutout_shape[0] <= 64)
                assert (0 <= cutout_shape[1] <= 64)
                assert (cutout_shape[2] == 2)

    def test_find_images_overlapping_spectrum(self):
        test_spectrum = "../../data/spectra/3126/spec-7330-56684-0434.fits"
        writer = h5u.SDSSCubeWriter(self.h5_file, self.cube_utils)
        with fitsio.FITS(test_spectrum) as hdul:
            writer.metadata = hdul[0].read_header()
        writer.find_images_overlapping_spectrum()

    def test_crop_cutout_to_image(self):
        top_left, top_right, bot_left, bot_right = [-32, -32], [32, -32], [-32, 32], [32, 32]
        e_top_left, e_top_right, e_bot_left, e_bot_right = [0, 0], [32, 0], [0, 32], [32, 32]
        h5u.SDSSCubeWriter.crop_cutout_to_image(top_left, top_right, bot_left, bot_right, np.array([2048, 1489]))
        assert (top_left == e_top_left)
        assert (top_right == e_top_right)
        assert (bot_left == e_bot_left)
        assert (top_right == e_top_right)

        top_left, top_right, bot_left, bot_right = [2047, 1488], [2111, 1488], [2047, 1552], [2111, 1552]
        e_top_left, e_top_right, e_bot_left, e_bot_right = [2047, 1488], [2047, 1488], [2047, 1488], [2047, 1488]
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

        top_left, top_right, bot_left, bot_right = [126, -24], [190, -24], [126, 40], [190, 40],
        e_top_left, e_top_right, e_bot_left, e_bot_right = [126, 0], [190, 0], [126, 40], [190, 40],
        h5u.SDSSCubeWriter.crop_cutout_to_image(top_left, top_right, bot_left, bot_right, np.array([2048, 1489]))
        assert (top_left == e_top_left)
        assert (top_right == e_top_right)
        assert (bot_left == e_bot_left)
        assert (top_right == e_top_right)

    def test_add_spectra_multiple(self):
        spectra_folder = "../../data/galaxy_small/spectra"
        test_spectra_small = "*.fits"
        writer = h5u.SDSSCubeWriter(self.h5_file, self.cube_utils)
        for spectrum in Path(spectra_folder).rglob(test_spectra_small):
            print("writing: %s" % spectrum)
            datasets = writer.ingest_spectrum(spectrum)
            assert (len(datasets) > 1)

    def test_float_compress(self):
        test_path = "../../data/images/301/4797/1/frame-g-004797-1-0019.fits.bz2"

        writer = h5u.SDSSCubeWriter(self.h5_file, self.cube_utils)
        writer.metadata, writer.data = writer.cube_utils.get_multiple_resolution_image(test_path, writer.IMG_MIN_RES)
        writer.file_name = os.path.basename(test_path)
        for img in writer.data:  # parsing 2D resolution
            img_data = np.dstack((img["flux_mean"], img["flux_sigma"]))
            float_compressed_data = writer.float_compress(img_data)
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
