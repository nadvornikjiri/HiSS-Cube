import numpy
from astropy import wcs
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.wcs.utils import skycoord_to_pixel


def get_boundary_coords(fits_header):
    """
    Gets boundary coords in ra, dec for an image.

    Parameters
    ----------
    fits_header Dictionary

    Returns     [(float, float)]
    -------

    """
    w = get_optimized_wcs(fits_header)
    coord_top_left = w.wcs_pix2world(0, 0, 0)
    coord_bot_left = w.wcs_pix2world(0, fits_header["NAXIS2"], 0)
    coord_top_right = w.wcs_pix2world(fits_header["NAXIS1"], 0, 0)
    coord_bot_right = w.wcs_pix2world(fits_header["NAXIS1"], fits_header["NAXIS2"], 0)
    return [coord_top_left, coord_bot_left, coord_top_right, coord_bot_right]


def get_optimized_wcs(image_fits_header):
    """
    Reads the WCS header and constructs the WCS object in an optimized way, see
    https://docs.astropy.org/en/stable/wcs/example_create_imaging.html for an idea how we do so.

    Parameters
    ----------
    image_fits_header  Dictionary

    Returns WCS object
    -------

    """
    w = wcs.WCS(naxis=2)
    w.wcs.crpix = [image_fits_header["CRPIX1"], image_fits_header["CRPIX2"]]
    w.wcs.cd = np.array([[image_fits_header["CD1_1"], image_fits_header["CD1_2"]],
                         [image_fits_header["CD2_1"], image_fits_header["CD2_2"]]])
    w.wcs.crval = [image_fits_header["CRVAL1"], image_fits_header["CRVAL2"]]
    w.wcs.ctype = [image_fits_header["CTYPE1"], image_fits_header["CTYPE2"]]
    return w


class NoCoverageFoundError(Exception):
    pass


def get_cutout_bounds(image_ds, res_idx, spectrum_fits_header, spectral_cutout_size):
    """
    Gets cutout bounds for an image dataset for a given resolution index (zoom) and a spectrum_fits_header where we get the location of that cutout.

    Parameters
    ----------
    image_ds                HDF5 dataset
    res_idx                 Resolution index = zoom factor
    spectrum_fits_header    Dictionary-like header of the spectrum, mostly copied from the FITS.

    Returns                 Numpy array shape (2,2)
    -------

    """
    w = get_optimized_wcs(image_ds.attrs)
    image_size = np.array((image_ds.attrs["NAXIS0"], image_ds.attrs["NAXIS1"]))
    return process_cutout_bounds(w, image_size, spectrum_fits_header, spectral_cutout_size, res_idx)


def process_cutout_bounds(w, image_size, spectrum_fits_header, spectral_cutout_size, res_idx=0):
    """
    Returns the process cutout_bounds for an image with a give w (WCS header), image_size, spectrum header and resolution index (zoom).
    Parameters
    ----------
    w                       FITS WCS initialized object.
    image_size              Numpy array, shape (2,)
    spectrum_fits_header    Dictionary-like header of the spectrum, mostly copied from the FITS.
    res_idx

    Returns                 Numpy array shape (2,2)
    -------

    """
    pixel_coords = np.array(skycoord_to_pixel(
        SkyCoord(ra=spectrum_fits_header["PLUG_RA"], dec=spectrum_fits_header["PLUG_DEC"], unit='deg'),
        w))
    if 0 <= pixel_coords[0] <= image_size[0] and 0 <= pixel_coords[1] <= image_size[1]:
        pixel_coords = (pixel_coords[0], pixel_coords[1])
        region_size = int(spectral_cutout_size / (2 ** res_idx))
        top_left = np.array((int(pixel_coords[0]) - (region_size / 2),
                             int(pixel_coords[1]) - (region_size / 2)), dtype=int)
        top_right = top_left + (region_size, 0)
        bot_left = top_left + (0, region_size)
        bot_right = top_left + (region_size, region_size)

        cutout_bounds = np.array([[top_left, top_right],
                                  [bot_left, bot_right]], dtype=int)
        return cutout_bounds
    else:
        raise NoCoverageFoundError


def is_cutout_whole(cutout_bounds, image_ds):
    return 0 <= cutout_bounds[0][0][0] <= cutout_bounds[0][1][0] <= image_ds.shape[1] and \
           0 <= cutout_bounds[1][0][0] <= cutout_bounds[1][1][0] <= image_ds.shape[1] and \
           0 <= cutout_bounds[0][0][1] <= cutout_bounds[0][1][1] <= image_ds.shape[0] and \
           0 <= cutout_bounds[1][0][1] <= cutout_bounds[1][1][1] <= image_ds.shape[0]
