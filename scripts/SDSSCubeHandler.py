import numpy as np
from astropy.coordinates import SkyCoord
from astropy.wcs import wcs
from astropy.wcs.utils import skycoord_to_pixel
from astropy.wcs.utils import pixel_to_skycoord

from scripts.astrometry import NoCoverageFoundError


class SDSSCubeHandler(object):
    def __init__(self, h5_file, cube_utils):
        self.cube_utils = cube_utils
        self.SPECTRAL_CUTOUT_SIZE = 64
        self.IMG_MIN_RES = 128
        self.SPEC_MIN_RES = 256
        self.IMG_SPAT_INDEX_ORDER = 8
        self.SPEC_SPAT_INDEX_ORDER = 14
        self.CHUNK_SIZE = (128, 128, 2)  # 128x128 pixels x (mean, var) tuples
        self.ORIG_CUBE_NAME = "orig_data_cube"
        self.NO_IMG_RESOLUTIONS = 5
        self.f = h5_file
        self.file_name = None
        self.fits_path = None
        self.data = None
        self.metadata = None

    def close_hdf5(self):
        self.f.close()

    def get_region_ref(self, res_idx, image_ds):
        cutout_bounds = self.get_cutout_bounds(image_ds, res_idx, self.metadata)
        region_ref = image_ds.regionref[cutout_bounds[0][1][1]:cutout_bounds[1][1][1],
                     cutout_bounds[1][0][0]:cutout_bounds[1][1][0]]
        cutout_shape = self.f[region_ref][region_ref].shape
        try:
            if not (0 <= cutout_shape[0] <= 64 and 0 <= cutout_shape[1] <= 64 and cutout_shape[2] == 2):
                raise NoCoverageFoundError
        except IndexError:
            raise NoCoverageFoundError
        return region_ref

    def get_cutout_bounds(self, image_ds, res_idx, spectrum_fits_header):
        full_res_link = image_ds.attrs.get("orig_res_link")
        if full_res_link:
            image_fits_header = self.f[full_res_link].attrs
        else:
            image_fits_header = image_ds.attrs
        w = self._get_optimized_wcs(image_fits_header)
        image_size = np.array((image_ds.attrs["NAXIS2"], image_ds.attrs["NAXIS1"]))
        pixel_coords = skycoord_to_pixel(
            SkyCoord(ra=spectrum_fits_header["PLUG_RA"], dec=spectrum_fits_header["PLUG_DEC"], unit='deg'), w)
        if 0 < pixel_coords[0] < image_size[0] and 0 < pixel_coords[1] < image_size[1]:
            pixel_coords = (pixel_coords[0] / (2 ** res_idx), pixel_coords[1] / (2 ** res_idx))
            region_size = int(self.SPECTRAL_CUTOUT_SIZE / (2 ** res_idx))
            top_left = np.array((int(pixel_coords[0]) - (region_size / 2),
                                 int(pixel_coords[1]) - (region_size / 2)), dtype=int)
            top_right = top_left + (region_size, 0)
            bot_left = top_left + (0, region_size)
            bot_right = top_left + (region_size, region_size)

            self.crop_cutout_to_image(top_left, top_right, bot_left, bot_right, image_size)
            cutout_bounds = np.array([[top_left, top_right],
                                      [bot_left, bot_right]], dtype=int)
            return cutout_bounds
        else:
            raise NoCoverageFoundError

    def _get_optimized_wcs(self, image_fits_header):
        w = wcs.WCS(naxis=2)
        w.wcs.crpix = [image_fits_header["CRPIX1"], image_fits_header["CRPIX2"]]
        w.wcs.cd = np.array([[image_fits_header["CD1_1"], image_fits_header["CD1_2"]],
                                [image_fits_header["CD2_1"], image_fits_header["CD2_2"]]])
        w.wcs.crval = [image_fits_header["CRVAL1"], image_fits_header["CRVAL2"]]
        w.wcs.ctype = [image_fits_header["CTYPE1"], image_fits_header["CTYPE2"]]
        return w

    @staticmethod
    def crop_cutout_to_image(top_left, top_right, bot_left, bot_right, image_size):
        image_indices = image_size - (1, 1)
        if top_left[0] < 0:
            top_left[0] = 0
        if top_left[1] < 0:
            top_left[1] = 0
        if top_right[0] > image_indices[0]:
            top_right[0] = image_indices[0]
        if top_right[1] < 0:
            top_right[1] = 0
        if bot_left[0] < 0:
            bot_left[0] = 0
        if bot_left[1] > image_indices[1]:
            bot_left[1] = image_indices[1]
        if bot_right[0] > image_indices[0]:
            bot_right[0] = image_indices[0]
        if bot_right[1] > image_indices[1]:
            bot_right[1] = image_indices[1]
