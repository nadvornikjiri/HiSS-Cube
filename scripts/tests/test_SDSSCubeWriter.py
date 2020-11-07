import os

import fitsio
from scripts import SDSSCubeWriter as h5u
from scripts import photometry as cu
import h5py
import pytest
import numpy as np
from pathlib import Path
import time
import warnings
from astropy.utils.exceptions import AstropyWarning
from tqdm.auto import tqdm
import timeit

from scripts.SDSSCubeHandler import is_cutout_whole

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
        image_path = "../../data/images/301/4797/1/frame-g-004797-1-0019.fits.bz2"

        writer = h5u.SDSSCubeWriter(self.h5_file, self.cube_utils)
        start_time = timeit.default_timer()
        writer.metadata, writer.data = writer.cube_utils.get_multiple_resolution_image(image_path, writer.IMG_MIN_RES)
        print(timeit.default_timer() - start_time)
        start_time = timeit.default_timer()
        writer.file_name = os.path.basename(image_path)
        res_grps = writer.create_image_index_tree()
        img_datasets = writer.create_img_datasets(res_grps)
        print(timeit.default_timer() - start_time)
        start_time = timeit.default_timer()
        writer.add_metadata(img_datasets)
        print(timeit.default_timer() - start_time)
        writer.f.flush()
        assert len(img_datasets) == 4

    @pytest.mark.usefixtures("truncate_test_file")
    def test_add_image2(self):
        test_path = "../../data/images_medium_ds/frame-u-004948-3-0199.fits.bz2"

        writer = h5u.SDSSCubeWriter(self.h5_file, self.cube_utils)
        h5_datasets = writer.ingest_image(test_path)
        assert len(h5_datasets) == 4

    def test_add_spectrum(self):
        test_path = "../../data/galaxy_small/spectra/spec-0411-51817-0119.fits"

        writer = h5u.SDSSCubeWriter(self.h5_file, self.cube_utils)

        start_time = timeit.default_timer()
        writer.metadata, writer.data = writer.cube_utils.get_multiple_resolution_spectrum(test_path,
                                                                                          writer.SPEC_MIN_RES)
        print(timeit.default_timer() - start_time)
        start_time = timeit.default_timer()
        writer.file_name = os.path.basename(test_path)
        res_grps = writer.create_spectrum_index_tree()
        spec_datasets = writer.create_spec_datasets(res_grps)
        print(timeit.default_timer() - start_time)
        start_time = timeit.default_timer()
        writer.add_metadata(spec_datasets)
        print(timeit.default_timer() - start_time)
        start_time = timeit.default_timer()
        writer.f.flush()
        writer.add_image_refs_to_spectra(spec_datasets)
        print(timeit.default_timer() - start_time)
        assert len(spec_datasets) == 4

    @pytest.mark.usefixtures("truncate_test_file")
    def test_add_image_multiple(self):
        # test_images = "../../data/images/301/2820/3/frame-*-002820-3-0122.fits.bz2"
        # test_images = "../../data/images_medium_ds"
        test_images = "../../data/galaxy_small/images"
        image_pattern = "*.fits.bz2"
        writer = h5u.SDSSCubeWriter(self.h5_file, self.cube_utils)
        image_paths = list(Path(test_images).rglob(image_pattern))
        for image in tqdm(image_paths, desc="Images completed: "):
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

    def test_add_spectra_multiple(self):
        spectra_folder = "../../data/galaxy_small/spectra"
        spectra_pattern = "*.fits"
        writer = h5u.SDSSCubeWriter(self.h5_file, self.cube_utils)
        spectra_paths = list(Path(spectra_folder).rglob(spectra_pattern))
        for spectrum in tqdm(spectra_paths, desc="Spectra completed: "):
            datasets = writer.ingest_spectrum(spectrum)
            assert (len(datasets) >= 1)

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
