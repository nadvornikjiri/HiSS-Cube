import numpy as np

from hisscube.H5Handler import H5Handler
from hisscube.utils.astrometry import get_optimized_wcs, get_cutout_bounds


class Processor(H5Handler):
    def __init__(self, h5_file=None, h5_path=None, timings_csv="timings.csv"):
        """
        Initializes the reader related properties, such as the array type for the exported dense cube.
        Parameters
        ----------
        h5_file
        cube_utils
        """
        super().__init__(h5_file=h5_file, h5_path=h5_path, timings_csv=timings_csv)

    def get_cutout_bounds_from_spectrum(self, image_ds, res_idx, spectrum_ds):
        orig_image_header = self.get_header(image_ds)
        orig_spectrum_header = self.get_header(spectrum_ds)
        time = self.get_time_from_image(orig_image_header)
        wl = image_ds.name.split('/')[-3]
        w = get_optimized_wcs(self.read_serialized_fits_header(image_ds))
        image_fits_header = self.read_serialized_fits_header(image_ds)
        cutout_bounds = get_cutout_bounds(image_fits_header, res_idx, orig_spectrum_header,
                                          self.IMAGE_CUTOUT_SIZE)
        return cutout_bounds, time, w, wl

    def get_header(self, image_ds):
        try:
            if image_ds.attrs["orig_res_link"]:
                orig_image_header = self.read_serialized_fits_header(self.f[image_ds.attrs["orig_res_link"]])
            else:
                orig_image_header = self.read_serialized_fits_header(image_ds)
        except KeyError:
            orig_image_header = self.read_serialized_fits_header(image_ds)
        return orig_image_header

    def get_cutout_pixel_coords(self, cutout_bounds, w):
        y = np.arange(cutout_bounds[0][1][1], cutout_bounds[1][1][1])
        x = np.arange(cutout_bounds[0][0][0], cutout_bounds[1][1][0])
        X, Y = np.meshgrid(x, y)
        ra, dec = w.wcs_pix2world(X, Y, 0)

        return ra, dec
