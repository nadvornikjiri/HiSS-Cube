import os
from pathlib import Path

import fitsio
import h5py
import numpy as np
import pytest
from tqdm.auto import tqdm

from hisscube.Photometry import Photometry
from hisscube.Writer import Writer
from hisscube.astrometry import is_cutout_whole

H5PATH = "../../data/processed/galaxy_small.h5"


@pytest.fixture(scope="session", autouse=False)
def truncate_test_file(request):
    writer = Writer(h5_path=H5PATH)
    writer.open_h5_file_serial(truncate=True)
    writer.close_h5_file()


@pytest.mark.incremental
class TestH5Writer:

    def setup_method(self, test_method):
        self.writer = Writer(h5_path=H5PATH)
        self.writer.open_h5_file_serial(truncate=False)
        self.h5_file = self.writer.f

    def teardown_method(self, test_method):
        self.h5_file.close()

    @pytest.mark.usefixtures("truncate_test_file")
    def test_add_image(self):
        test_path = "../../data/raw/images_medium_ds/frame-u-004948-3-0199.fits.bz2"

        h5_datasets = self.writer.ingest_image(test_path)
        assert len(h5_datasets) == self.writer.config.getint("Handler", "IMG_ZOOM_CNT")

    def test_add_spectrum(self):
        test_path = "../../data/raw/galaxy_small/spectra/spec-0411-51817-0119.fits"

        writer = Writer(self.h5_file)

        h5_datasets = writer.ingest_spectrum(test_path)
        assert len(h5_datasets) == writer.config.getint("Handler", "SPEC_ZOOM_CNT")



    @pytest.mark.usefixtures("truncate_test_file")
    def test_add_image_multiple(self):
        #test_images = "../../data/images/301/2820/3"
        # test_images = "../../data/images_medium_ds"
        test_images = "../../data/raw/galaxy_small/images"
        image_pattern = "frame-*-004136-*-0129.fits"
        image_paths = list(Path(test_images).rglob(image_pattern))
        for image in tqdm(image_paths, desc="Images completed: "):
            self.writer.ingest_image(image)

    def test_add_spec_refs(self):
        test_spectrum = "../../data/raw/galaxy_small/spectra/spec-0411-51817-0119.fits"
        spec_datasets = self.writer.ingest_spectrum(test_spectrum)
        self.writer.add_image_refs_to_spectra(spec_datasets)
        assert bool(spec_datasets[0].parent.parent.parent["image_cutouts_0"][0])
        for spec_dataset in spec_datasets:
            for zoom in range(self.writer.config.getint("Handler", "SPEC_ZOOM_CNT")):
                for cutout in spec_dataset.parent.parent.parent["image_cutouts_%d" % zoom]:
                    if not cutout:
                        break
                    cutout_shape = self.h5_file[cutout][cutout].shape
                    assert (0 <= cutout_shape[0] <= 64)
                    assert (0 <= cutout_shape[1] <= 64)
                    assert (cutout_shape[2] == 2)

    def test_add_spec_refs2(self):
        test_spectrum = "../../data/raw/spectra_medium_ds/spec-4238-55455-0037.fits"
        spec_datasets = self.writer.ingest_spectrum(test_spectrum)
        spec_datasets = self.writer.add_image_refs_to_spectra(spec_datasets)
        for spec_dataset in spec_datasets:
            for zoom in range(self.writer.config.getint("Handler", "SPEC_ZOOM_CNT")):
                for cutout in spec_dataset.parent.parent.parent["image_cutouts_%d" % zoom]:
                    if not cutout:
                        break
                    cutout_shape = self.h5_file[cutout][cutout].shape
                    assert (0 <= cutout_shape[0] <= 64)
                    assert (0 <= cutout_shape[1] <= 64)
                    assert (cutout_shape[2] == 2)

    def test_find_images_overlapping_spectrum(self):
        test_spectrum = "../../data/raw/galaxy_small/spectra/spec-0411-51817-0119.fits"
        with fitsio.FITS(test_spectrum) as hdul:
            self.writer.metadata = hdul[0].read_header()
        for image in self.writer.find_images_overlapping_spectrum():
            assert True

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
        spectra_folder = "../../data/raw/galaxy_small/spectra"
        spectra_pattern = "*.fits"
        spectra_paths = list(Path(spectra_folder).rglob(spectra_pattern))
        for spectrum in tqdm(spectra_paths, desc="Spectra completed: "):
            datasets = self.writer.ingest_spectrum(spectrum)
            assert (len(datasets) >= 1)

    def test_add_spec_refs_multiple(self):
        self.writer.add_image_refs(self.h5_file)
        assert True



    def test_rebin(self):
        spectra_folder = "../../data/raw/galaxy_small/spectra"
        spectra_pattern = "*.fits"
        writer = Writer(self.h5_file)
        spectra_paths = list(Path(spectra_folder).rglob(spectra_pattern))
        for spec_path in tqdm(spectra_paths, desc="Spectra completed: "):
            metadata, data = writer.cube_utils.get_multiple_resolution_spectrum(
                spec_path,
                writer.config.getint("Handler", "SPEC_ZOOM_CNT"),
                apply_rebin=writer.config.getboolean("Preprocessing", "APPLY_REBIN"),
                rebin_min=writer.config.getfloat("Preprocessing", "REBIN_MIN"),
                rebin_max=writer.config.getfloat("Preprocessing", "REBIN_MAX"),
                rebin_samples=writer.config.getint("Preprocessing", "REBIN_SAMPLES"),
                apply_transmission=writer.config.getboolean("Preprocessing", "APPLY_TRANSMISSION_CURVE"))
            assert (data[0]['flux_mean'].shape[0] == 4620)

    def test_write_dense_cube(self):
        writer = Writer(self.h5_file)
        writer.create_dense_cube()
        assert True

    @pytest.mark.skip("Annoyingly long run.")
    def test_float_compress(self):
        test_path = "../../data/raw/images/301/4797/1/frame-g-004797-1-0019.fits.bz2"

        writer = Writer(self.h5_file)
        writer.metadata, writer.data = writer.cube_utils.get_multiple_resolution_image(test_path,
                                                                                       writer.config.getint("Handler",
                                                                                                            "IMG_ZOOM_CNT"))
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
