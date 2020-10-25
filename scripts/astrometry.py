from astropy import wcs
import numpy as np


def get_boundary_coords(fits_header):
    w = get_optimized_wcs(fits_header)
    coord_top_left = w.wcs_pix2world(0, 0, 0)
    coord_bot_left = w.wcs_pix2world(0, fits_header["NAXIS2"], 0)
    coord_top_right = w.wcs_pix2world(fits_header["NAXIS1"], 0, 0)
    coord_bot_right = w.wcs_pix2world(fits_header["NAXIS1"], fits_header["NAXIS2"], 0)
    return [coord_top_left, coord_bot_left, coord_top_right, coord_bot_right]

def get_optimized_wcs(image_fits_header):
    w = wcs.WCS(naxis=2)
    w.wcs.crpix = [image_fits_header["CRPIX1"], image_fits_header["CRPIX2"]]
    w.wcs.cd = np.array([[image_fits_header["CD1_1"], image_fits_header["CD1_2"]],
                         [image_fits_header["CD2_1"], image_fits_header["CD2_2"]]])
    w.wcs.crval = [image_fits_header["CRVAL1"], image_fits_header["CRVAL2"]]
    w.wcs.ctype = [image_fits_header["CTYPE1"], image_fits_header["CTYPE2"]]
    return w


class NoCoverageFoundError(Exception):
    pass
