import os
from urllib.parse import urljoin

import pytest
from astropy.samp import SAMPIntegratedClient

from hisscube.processors.fits_cube_visualization import FITSProcessor
from hisscube.utils.config import Config
from hisscube.utils.photometry import Photometry


@pytest.mark.skip(reason="Deprecated.")
class TestFITSCubeProcessor:

    def setup_method(self, test_method):
        spectra_path = "../../data/raw/galaxy_small/spectra"
        image_path = "../../data/raw/galaxy_small/images"
        self.processor = FITSProcessor(Config(), Photometry(), spectra_path, image_path,
                                       image_regex="frame-*-004136-*-0129.fits")
        self.resolution = 0

    def test_write_FITS(self):
        self.output_path = "../../results/output.fits"
        data = self.processor.get_spectral_cube_from_orig_for_res(self.resolution)
        self.processor.write_FITS(self.output_path)
        assert data.shape == (267219,)
