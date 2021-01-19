from hisscube.astrometry import NoCoverageFoundError, get_cutout_bounds, is_cutout_whole
import configparser
import pathlib
from ast import literal_eval as make_tuple


class H5Handler(object):
    def __init__(self, h5_file=None, cube_utils=None):
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
        self.INCLUDE_ADDITIONAL_METADATA = self.config.getboolean("Handler", "INCLUDE_ADDITIONAL_METADATA")
        self.INIT_ARRAY_SIZE = int(self.config["Handler"]["INIT_ARRAY_SIZE"])
        self.FITS_MEM_MAP = self.config.getboolean("Handler", "FITS_MEM_MAP")

        self.cube_utils = cube_utils
        self.f = h5_file
        self.h5_path = None
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
        cutout_bounds = get_cutout_bounds(image_ds, res_idx, self.metadata)
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
