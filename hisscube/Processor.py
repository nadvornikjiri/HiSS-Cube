import logging

import numpy as np

from hisscube.H5Handler import H5Handler
from hisscube.astrometry import get_optimized_wcs, get_cutout_bounds


class Processor(H5Handler):
    def __init__(self, h5_file, cube_utils):
        """
        Initializes the reader related properties, such as the array type for the exported dense cube.
        Parameters
        ----------
        h5_file
        cube_utils
        """
        super().__init__(h5_file, cube_utils)

        self.logger = logging.getLogger(self.__class__.__name__)
        self.OUTPUT_HEAL_ORDER = int(self.config["Reader"]["OUTPUT_HEAL_ORDER"])

    def get_cutout_bounds_from_spectrum(self, image_ds, res_idx, spectrum_ds):
        orig_image_header = self.get_header(image_ds)
        orig_spectrum_header = self.get_header(spectrum_ds)
        time = self.get_time_from_image(orig_image_header)
        wl = image_ds.name.split('/')[-3]
        w = get_optimized_wcs(image_ds.attrs)
        cutout_bounds = get_cutout_bounds(image_ds, res_idx, orig_spectrum_header,
                                          self.config.getint("Handler", "SPECTRAL_CUTOUT_SIZE"))
        return cutout_bounds, time, w, wl

    def get_header(self, image_ds):
        try:
            if image_ds.attrs["orig_res_link"]:
                orig_image_header = self.f[image_ds.attrs["orig_res_link"]].attrs
            else:
                orig_image_header = image_ds.attrs
        except KeyError:
            orig_image_header = image_ds.attrs
        return orig_image_header

    def get_cutout_pixel_coords(self, cutout_bounds, w):
        y = np.arange(cutout_bounds[0][1][1], cutout_bounds[1][1][1])
        x = np.arange(cutout_bounds[0][0][0], cutout_bounds[1][1][0])
        X, Y = np.meshgrid(x, y)
        ra, dec = w.wcs_pix2world(X, Y, 0)

        return ra, dec
