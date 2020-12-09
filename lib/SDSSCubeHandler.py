import numpy as np
from astropy.coordinates import SkyCoord
from astropy.wcs.utils import skycoord_to_pixel

from lib.astrometry import NoCoverageFoundError, get_optimized_wcs
import configparser
import pathlib
from ast import literal_eval as make_tuple


def is_cutout_whole(cutout_bounds, image_ds):
    return 0 <= cutout_bounds[0][0][0] <= cutout_bounds[0][1][0] <= image_ds.shape[1] and \
           0 <= cutout_bounds[1][0][0] <= cutout_bounds[1][1][0] <= image_ds.shape[1] and \
           0 <= cutout_bounds[0][0][1] <= cutout_bounds[0][1][1] <= image_ds.shape[0] and \
           0 <= cutout_bounds[1][0][1] <= cutout_bounds[1][1][1] <= image_ds.shape[0]


class SDSSCubeHandler(object):
    def __init__(self, h5_file, cube_utils):
        """
        Initialize contains configuration relevant to both HDF5 Reader and Writer.
        Parameters
        ----------
        h5_file     Opened HDF5 file object
        cube_utils  Loaded cube_utils, containing mainly photometry-related constants needed for preprocessing.
        """
        lib_path = pathlib.Path(__file__).parent.absolute()
        self.config = configparser.ConfigParser(allow_no_value=True)
        self.config.read("%s/config.ini" % lib_path)
        self.SPECTRAL_CUTOUT_SIZE = int(self.config["Handler"]["SPECTRAL_CUTOUT_SIZE"])
        self.IMG_MIN_RES = int(self.config["Handler"]["IMG_MIN_RES"])
        self.SPEC_MIN_RES = int(self.config["Handler"]["SPEC_MIN_RES"])
        self.IMG_SPAT_INDEX_ORDER = int(self.config["Handler"]["IMG_SPAT_INDEX_ORDER"])
        self.SPEC_SPAT_INDEX_ORDER = int(self.config["Handler"]["SPEC_SPAT_INDEX_ORDER"])
        self.CHUNK_SIZE = make_tuple(self.config["Handler"]["CHUNK_SIZE"])
        self.ORIG_CUBE_NAME = self.config["Handler"]["ORIG_CUBE_NAME"]
        self.DENSE_CUBE_NAME = self.config["Handler"]["DENSE_CUBE_NAME"]
        self.NO_IMG_RESOLUTIONS = int(self.config["Handler"]["NO_IMG_RESOLUTIONS"])
        self.INCLUDE_ADDITIONAL_METADATA = bool(self.config["Handler"]["INCLUDE_ADDITIONAL_METADATA"])
        self.INIT_ARRAY_SIZE = int(self.config["Handler"]["INIT_ARRAY_SIZE"])

        self.cube_utils = cube_utils
        self.f = h5_file
        self.file_name = None
        self.fits_path = None
        self.data = None
        self.metadata = None

    def close_hdf5(self):
        self.f.close()

    def get_region_ref(self, res_idx, image_ds):
        """
        Gets the region reference for a given resolution from an image_ds.

        Parameters
        ----------
        res_idx     Resolution index = zoom factor, e.g., 0, 1, 2, ...
        image_ds    HDF5 dataset

        Returns     HDF5 region reference
        -------

        """
        cutout_bounds = self.get_cutout_bounds(image_ds, res_idx, self.metadata)
        if not is_cutout_whole(cutout_bounds, image_ds):
            raise NoCoverageFoundError
        region_ref = image_ds.regionref[cutout_bounds[0][1][1]:cutout_bounds[1][1][1],
                     cutout_bounds[1][0][0]:cutout_bounds[1][1][0]]
        cutout_shape = self.f[region_ref][region_ref].shape
        try:
            if not (0 <= cutout_shape[0] <= (64 / 2 ** res_idx) and
                    0 <= cutout_shape[1] <= (64 / 2 ** res_idx) and
                    cutout_shape[2] == 2):
                raise NoCoverageFoundError
        except IndexError:
            raise NoCoverageFoundError
        return region_ref

    def get_cutout_bounds(self, image_ds, res_idx, spectrum_fits_header):
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
        return self.process_cutout_bounds(w, image_size, spectrum_fits_header, res_idx)

    def process_cutout_bounds(self, w, image_size, spectrum_fits_header, res_idx=0):
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
            region_size = int(self.SPECTRAL_CUTOUT_SIZE / (2 ** res_idx))
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
