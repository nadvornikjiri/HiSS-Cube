import math

import healpy as hp
import numpy as np
from astropy import wcs
from astropy.coordinates import SkyCoord
from astropy.wcs.utils import skycoord_to_pixel
from numpy import arange


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


def get_image_center_coords(fits_header):
    return fits_header["CRVAL1"], fits_header["CRVAL2"]


def get_spectrum_center_coords(fits_header):
    return fits_header["PLUG_RA"], fits_header["PLUG_DEC"]


def get_optimized_wcs(image_fits_header):
    """
    Reads the WCS header and constructs the WCS object in an optimized way, see
    https://docs.astropy.org/en/stable/wcs/example_create_imaging.html for an idea how we do so.

    Parameters
    ----------
    fits_header  Dictionary

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


def get_cutout_bounds(image_fits_header, res_idx, spectrum_fits_header, cutout_size):
    """
    Gets cutout bounds for an image dataset for a given resolution index (zoom) and a spectrum_fits_header where we get the location of that cutout.

    Parameters
    ----------
    ds                HDF5 dataset
    image_zoom                 Resolution index = zoom factor
    spectrum_fits_header    Dictionary-like header of the spectrum, mostly copied from the FITS.

    Returns                 Numpy array shape (2,2)
    -------

    """
    w = get_optimized_wcs(image_fits_header)
    image_size = np.array((image_fits_header["NAXIS1"], image_fits_header["NAXIS2"]))
    return process_cutout_bounds(w, image_size, spectrum_fits_header, cutout_size, res_idx)


def process_cutout_bounds(w, image_size, spectrum_fits_header, cutout_size, res_idx=0):
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
        region_size = int(cutout_size / (2 ** res_idx))
        top_left = np.array((int(pixel_coords[0]) - (region_size / 2),
                             int(pixel_coords[1]) - (region_size / 2)), dtype=int)
        top_right = top_left + (region_size, 0)
        bot_left = top_left + (0, region_size)
        bot_right = top_left + (region_size, region_size)

        cutout_bounds = np.array([[top_left, top_right],
                                  [bot_left, bot_right]], dtype=int)
        return cutout_bounds
    else:
        raise NoCoverageFoundError("The spectrum pixel is not within image bounds.")


def is_cutout_whole(cutout_bounds, image_ds):
    return 0 <= cutout_bounds[0][0][0] <= cutout_bounds[0][1][0] <= image_ds.shape[1] and \
           0 <= cutout_bounds[1][0][0] <= cutout_bounds[1][1][0] <= image_ds.shape[1] and \
           0 <= cutout_bounds[0][0][1] <= cutout_bounds[0][1][1] <= image_ds.shape[0] and \
           0 <= cutout_bounds[1][0][1] <= cutout_bounds[1][1][1] <= image_ds.shape[0]


def get_potential_overlapping_image_spatial_paths(fits_header, radius_arcmin, image_index_depth, image_size_x=None,
                                                  image_size_y=None):
    nsides = 2 ** arange(image_index_depth)
    reference_nside = nsides[-1]
    fact = 2 ** image_index_depth
    pix_ids = get_overlapping_healpix_pixel_ids(fits_header, reference_nside, fact, radius_arcmin, image_size_x,
                                                image_size_y)
    paths = []
    for ipix in pix_ids:
        path = str(ipix)
        for nside in reversed(nsides[:-1]):
            ipix_vec = hp.pix2vec(nside * 2, ipix, nest=True)
            ipix = hp.vec2pix(nside, ipix_vec[0], ipix_vec[1], ipix_vec[2], nest=True)
            path = "%s/%s" % (ipix, path)
        paths.append(path)
    return paths


def get_overlapping_healpix_pixel_ids(fits_header, nside, fact, radius_arcmin, image_size_x=None, image_size_y=None):
    spec_ra = fits_header["PLUG_RA"]
    spec_dec = fits_header["PLUG_DEC"]
    vec = hp.ang2vec(spec_ra, spec_dec, lonlat=True)
    if not (image_size_x and image_size_y):
        radius_rad = radius_arcmin * math.pi / (60 * 180)
        pix_ids = hp.query_disc(nside, vec, inclusive=True, fact=fact, radius=radius_rad, nest=True)
    else:
        center_ra = SkyCoord(spec_ra, spec_dec, unit='deg')
        potential_image_region = RectangleSkyRegion

        half_image_x = image_size_x / 60 / 2
        half_image_y = image_size_y / 60 / 2
        ra_poly, dec_poly = (np.array())
        bound_0 = (spec_ra + half_image_y, spec_dec - half_image_x)
        vec_0 = hp.ang2vec(bound_0[0], bound_0[1], lonlat=True)
        bound_1 = (spec_ra + half_image_y, spec_dec + half_image_x)
        vec_1 = hp.ang2vec(bound_1[0], bound_1[1], lonlat=True)
        bound_2 = (spec_ra - half_image_y, spec_dec - half_image_x)
        vec_2 = hp.ang2vec(bound_2[0], bound_2[1], lonlat=True)
        bound_3 = (spec_ra - half_image_y, spec_dec + half_image_x)
        vec_3 = hp.ang2vec(bound_3[0], bound_3[1], lonlat=True)
        pix_ids = hp.query_polygon(nside, [vec_0, vec_1, vec_2, vec_3], inclusive=True, fact=fact)
    return pix_ids


def get_image_lower_res_wcs(orig_image_fits_header, image_fits_header, res_idx=0):
    """
    Modifies the FITS WCS parameters for lower resolutions of the image so it is still correct.
    Parameters
    ----------
    ds      HDF5 dataset
    image_zoom int

    Returns
    -------

    """
    w = get_optimized_wcs(orig_image_fits_header)
    w.wcs.crpix /= 2 ** res_idx  # shift center of the image
    w.wcs.cd *= 2 ** res_idx  # change the pixel scale
    image_fits_header["CRPIX1"], image_fits_header["CRPIX2"] = w.wcs.crpix
    [[image_fits_header["CD1_1"], image_fits_header["CD1_2"]],
     [image_fits_header["CD2_1"], image_fits_header["CD2_2"]]] = w.wcs.cd
    image_fits_header["CRVAL1"], image_fits_header["CRVAL2"] = w.wcs.crval
    image_fits_header["CTYPE1"], image_fits_header["CTYPE2"] = w.wcs.ctype
    return image_fits_header


def get_heal_path_from_coords(metadata, config, ra=None, dec=None, order=None):
    if ra is None and dec is None:
        ra = metadata["PLUG_RA"]
        dec = metadata["PLUG_DEC"]
    if order is None:
        order = config.IMG_SPAT_INDEX_ORDER
    pixel_IDs = hp.ang2pix(hp.order2nside(np.arange(order)),
                           ra,
                           dec,
                           nest=True,
                           lonlat=True)
    heal_path = "/".join(str(pixel_ID) for pixel_ID in pixel_IDs)
    absolute_path = "%s/%s" % (config.ORIG_CUBE_NAME, heal_path)
    return absolute_path


def get_cutout_pixel_coords(cutout_bounds, w):
    y = np.arange(cutout_bounds[0][1][1], cutout_bounds[1][1][1])
    x = np.arange(cutout_bounds[0][0][0], cutout_bounds[1][1][0])
    X, Y = np.meshgrid(x, y)
    ra, dec = w.wcs_pix2world(X, Y, 0)

    return ra, dec
