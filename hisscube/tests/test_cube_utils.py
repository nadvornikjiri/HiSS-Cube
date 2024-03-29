from hisscube.utils import photometry as utils
import numpy as np

from hisscube.utils.config import Config


class TestCubeUtils:

    def setup_method(self, test_method):
        self.cube_utils = utils.Photometry(Config())

    def test_get_spectrum_lower_resolution(self):
        test_spectrum = "../../data/raw/spectra/spec-4500-55543-0331.fits"
        expected_spectra_resolutions = [4620, 2310, 1155, 577, 288]
        SPEC_ZOOM_CNT = 4

        fits_header, multiple_res_cube = self.cube_utils.get_multiple_resolution_spectrum(test_spectrum, SPEC_ZOOM_CNT,
                                                                                          apply_transmission=True)
        assert (len(multiple_res_cube) == len(expected_spectra_resolutions))
        for i, res in enumerate(expected_spectra_resolutions):
            assert (multiple_res_cube[i]["zoom_idx"] == expected_spectra_resolutions[i])

    def test__get_image_with_errors(self):
        test_image = "../../data/raw/images/301/4797/1/frame-g-004797-1-0019.fits.bz2"
        fits_header, img, img_err = self.cube_utils._get_image_with_errors(test_image)
        assert (img.shape[1] == 2048 and img.shape[0] == 1489)
        assert (img.shape == img_err.shape)
        assert (np.all(0 <= err <= 1 for err in img_err))

    def test_get_image_lower_resolution(self):
        test_image = "../../data/raw/images/301/4797/1/frame-g-004797-1-0019.fits.bz2"
        expected_image_resolutions = [(2048, 1489), (1024, 744), (512, 372), (256, 186), (128, 93)]
        IMG_ZOOM_CNT = 4
        fits_header, multiple_res_cube = self.cube_utils.get_multiple_resolution_image(test_image, IMG_ZOOM_CNT)
        assert (len(multiple_res_cube) == len(expected_image_resolutions))
        for i, res in enumerate(expected_image_resolutions):
            assert (multiple_res_cube[i]["zoom_idx"] == expected_image_resolutions[i])
            assert (multiple_res_cube[i]["flux_mean"].shape[0] == expected_image_resolutions[i][1])
            assert (multiple_res_cube[i]["flux_mean"].shape[1] == expected_image_resolutions[i][0])
            assert (multiple_res_cube[i]["flux_sigma"].shape[0] == expected_image_resolutions[i][1])
            assert (multiple_res_cube[i]["flux_sigma"].shape[1] == expected_image_resolutions[i][0])

    def test_merge_transmission_curves_max(self):
        merged_curves = self.cube_utils._merge_transmission_curves_max(self.cube_utils.transmission_curves)

        assert(len(merged_curves) == 331)