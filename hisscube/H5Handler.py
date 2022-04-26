import configparser
import csv
import logging
import pathlib
from datetime import datetime
from math import log

import h5py
import healpy as hp
import numpy as np
import ujson
from astropy.time import Time

from hisscube.astrometry import NoCoverageFoundError, get_cutout_bounds, is_cutout_whole, get_optimized_wcs
from hisscube import Photometry as cu


class H5Handler(object):
    def __init__(self, h5_file=None, h5_path=None, timings_csv="timings.csv"):
        """
        Initialize contains configuration relevant to both HDF5 Reader and Writer.
        Parameters
        ----------
        h5_file     Opened HDF5 file object
        cube_utils  Loaded cube_utils, containing mainly photometry-related constants needed for preprocessing.
        """
        self.spec_cnt = 0
        self.img_cnt = 0
        lib_path = pathlib.Path(__file__).parent.absolute()
        self.config = configparser.ConfigParser(allow_no_value=True)
        self.config.read("%s/config.ini" % lib_path)
        self.parse_config()
        # utils
        lib_path = pathlib.Path(__file__).parent.absolute()
        cube_utils = self.cube_utils = cu.Photometry("%s/../config/SDSS_Bands" % lib_path,
                                                     "%s/../config/ccd_gain.tsv" % lib_path,
                                                     "%s/../config/ccd_dark_variance.tsv" % lib_path)
        self.ingest_type = None
        self.spectrum_length = None
        self.image_path_list = []
        self.spectra_path_list = []

        self.cube_utils = cube_utils
        self.f = h5_file
        self.h5_path = h5_path
        self.file_name = None
        self.fits_path = None
        self.data = None
        self.metadata = None

        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(self.LOG_LEVEL)
        self.grp_cnt = 0

    def close_h5_file(self):
        self.f.flush()
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
        image_fits_header = self.read_serialized_fits_header(image_ds)
        cutout_bounds = get_cutout_bounds(image_fits_header, res_idx, self.metadata,
                                          self.IMAGE_CUTOUT_SIZE)
        if not is_cutout_whole(cutout_bounds, image_ds):
            raise NoCoverageFoundError("Cutout not whole.")
        region_ref = image_ds.regionref[cutout_bounds[0][1][1]:cutout_bounds[1][1][1],
                     cutout_bounds[1][0][0]:cutout_bounds[1][1][0]]
        cutout_shape = self.f[region_ref][region_ref].shape
        try:
            if not (0 <= cutout_shape[0] <= (64 / 2 ** res_idx) and
                    0 <= cutout_shape[1] <= (64 / 2 ** res_idx) and
                    cutout_shape[2] == 2):
                raise NoCoverageFoundError("Cutout not in correct shape.")
        except IndexError:
            raise NoCoverageFoundError("IndexError")
        return region_ref

    def require_raw_cube_grp(self):
        return self.require_group(self.f, self.ORIG_CUBE_NAME)

    def require_spatial_grp(self, order, prev, coord):
        """
        Returns the HEALPix group structure.
        Parameters
        ----------
        order   int
        prev    HDF5 group
        coord   (float, float)

        Returns
        -------

        """
        nside = 2 ** order
        healID = hp.ang2pix(nside, coord[0], coord[1], lonlat=True, nest=True)
        grp = self.require_group(prev, str(healID))  # TODO optimize to 8-byte string?
        self.set_attr(grp, "type", "spatial")
        return grp

    @staticmethod
    def add_hard_links(parent_groups, child_groups):
        for parent in parent_groups:
            for child_name in child_groups:
                if not child_name in parent:
                    parent[child_name] = child_groups[child_name]

    def require_res_grps(self, parent_grp):
        res_grps = []
        if self.ingest_type == "image":  ##if I'm an image
            x_lower_res = int(self.metadata["NAXIS1"])
            y_lower_res = int(self.metadata["NAXIS2"])
            for res_zoom in range(self.IMG_ZOOM_CNT):
                res_grp_name = str((x_lower_res, y_lower_res))
                grp = self.require_group(parent_grp, res_grp_name)
                self.set_attr(grp, "type", "resolution")
                self.set_attr(grp, "res_zoom", res_zoom)
                res_grps.append(grp)
                x_lower_res = int(x_lower_res / 2)
                y_lower_res = int(y_lower_res / 2)
        else:
            x_lower_res = int(self.spectrum_length)
            for res_zoom in range(self.SPEC_ZOOM_CNT):
                res_grp_name = str(x_lower_res)
                grp = self.require_group(parent_grp, res_grp_name)
                self.set_attr(grp, "type", "resolution")
                self.set_attr(grp, "res_zoom", res_zoom)
                res_grps.append(grp)
                x_lower_res = int(x_lower_res / 2)
        return res_grps

    def add_metadata(self, datasets):
        """
        Adds metadata to the HDF5 data sets of the same image or spectrum in multiple resolutions. It also modifies the
        metadata for image where needed and adds the COMMENT and HISTORY attributes as datasets for optimization
        purposes.
        Parameters
        ----------
        datasets    [HDF5 Datasets]

        Returns
        -------

        """
        unicode_dt = h5py.special_dtype(vlen=str)
        fits_header = dict(self.metadata)
        for res_idx, ds in enumerate(datasets):
            if res_idx > 0:
                self.set_attr_ref(ds, "orig_res_link", datasets[0])
                if self.get_attr(ds, "mime-type") == "image":
                    fits_header = self.write_image_lower_res_wcs(fits_header, res_idx)
            naxis = len(self.get_shape(ds))
            fits_header["NAXIS"] = naxis
            for axis in range(naxis):
                fits_header["NAXIS%d" % (axis)] = self.get_shape(ds)[axis]
            self.write_serialized_fits_header(ds, fits_header)

    def write_image_lower_res_wcs(self, image_fits_header, res_idx=0):
        """
        Modifies the FITS WCS parameters for lower resolutions of the image so it is still correct.
        Parameters
        ----------
        ds      HDF5 dataset
        res_idx int

        Returns
        -------

        """
        w = get_optimized_wcs(self.metadata)
        w.wcs.crpix /= 2 ** res_idx  # shift center of the image
        w.wcs.cd *= 2 ** res_idx  # change the pixel scale
        image_fits_header["CRPIX1"], image_fits_header["CRPIX2"] = w.wcs.crpix
        [[image_fits_header["CD1_1"], image_fits_header["CD1_2"]],
         [image_fits_header["CD2_1"], image_fits_header["CD2_2"]]] = w.wcs.cd
        image_fits_header["CRVAL1"], image_fits_header["CRVAL2"] = w.wcs.crval
        image_fits_header["CTYPE1"], image_fits_header["CTYPE2"] = w.wcs.ctype
        return image_fits_header

    def get_heal_path_from_coords(self, ra=None, dec=None, order=None):
        if ra is None and dec is None:
            ra = self.metadata["PLUG_RA"]
            dec = self.metadata["PLUG_DEC"]
        if order is None:
            order = self.IMG_SPAT_INDEX_ORDER
        pixel_IDs = hp.ang2pix(hp.order2nside(np.arange(order)),
                               ra,
                               dec,
                               nest=True,
                               lonlat=True)
        heal_path = "/".join(str(pixel_ID) for pixel_ID in pixel_IDs)
        absolute_path = "%s/%s" % (self.ORIG_CUBE_NAME, heal_path)
        return absolute_path

    def require_group(self, parent_grp, name, track_order=False):
        if not name in parent_grp:
            self.grp_cnt += 1
            return parent_grp.create_group(name, track_order=track_order)
        grp = parent_grp[name]
        return grp

    @staticmethod
    def float_compress(data, ndig=10):
        """
        Makes the data more compressible by zeroing bits of the mantissa.  The method is rewritten from the SDSS IDL
        variant http://www.sdss3.org/dr8/software/idlutils_doc.php#FLOATCOMPRESS.

        This function does not compress the data in an array, but fills
        unnecessary digits of the IEEE floating point representation with
        zeros.  This makes the data more compressible by standard
        compression routines such as compress or gzip.

        The default is to retain 10 binary digits instead of the usual 23
        bits (or 52 bits for double precision), introducing a fractional
        error strictly less than 1/1024).  This is adequate for most
        astronomical images, and results in images that compress a factor
        of 2-4 with gzip.

        Parameters
        ----------
        data    numpy array, type float32 or float64
        ndig    number of binary significant digits to keep

        Returns
        -------

        """
        data = data.astype(np.float32)
        wzer = np.where((data == 0) | (data == np.Inf))

        # replace zeros and infinite values with ones temporarily
        if len(wzer) > 0:
            temp = data[wzer]
            data[wzer] = 1.

        # compute log base 2
        log2 = np.ceil(np.log(np.abs(data)) / log(2.))  # exponent part

        mant = np.round(data / 2.0 ** (log2 - ndig)) / (2.0 ** ndig)  # mantissa, truncated
        out = mant * 2.0 ** log2  # multiple 2^exponent back in

        if len(wzer) > 0:
            out[wzer] = temp

        return out

    def get_time_from_image(self, orig_image_header):
        time_attr = orig_image_header["DATE-OBS"]
        try:
            time = Time(time_attr, format='isot', scale='tai').mjd
        except ValueError:
            time = Time(datetime.strptime(time_attr, "%d/%m/%y")).mjd
        return time

    def get_time_from_spectrum(self, spec_header):
        try:
            time = spec_header["TAI"]
        except KeyError:
            time = spec_header["MJD"]
        return time

    def get_property_list(self, dataset_shape):
        """
        Creates the property list so it is compatible for parallel file write and reading.

        Parameters
        ----------
        dataset_shape

        Returns
        -------

        """
        dataset_type = h5py.h5t.py_create(np.dtype('f4'))
        dcpl = h5py.h5p.create(h5py.h5p.DATASET_CREATE)
        dcpl.set_alloc_time(h5py.h5d.ALLOC_TIME_EARLY)
        dcpl.set_fill_time(h5py.h5d.FILL_TIME_NEVER)
        space = h5py.h5s.create_simple(dataset_shape)
        if self.COMPRESSION:
            dcpl.set_deflate(self.COMPRESSION_OPTS)
        if self.SHUFFLE:
            dcpl.set_shuffle()
        return dcpl, space, dataset_type

    @staticmethod
    def set_attr(obj, key, val):
        obj.attrs[key] = val

    @staticmethod
    def set_attr_ref(obj, key, obj2):
        obj.attrs[key] = obj2.ref

    @staticmethod
    def get_attr(ds, key):
        return ds.attrs[key]

    @staticmethod
    def write_serialized_fits_header(ds, attrs_dict):
        ds.attrs["serialized_header"] = ujson.dumps(attrs_dict)

    @staticmethod
    def read_serialized_fits_header(ds):
        return ujson.loads(ds.attrs["serialized_header"])

    @staticmethod
    def require_dataset(grp, name, shape, dtype):
        grp.require_dataset(name, shape, dtype)

    @staticmethod
    def get_shape(ds):
        return ds.shape

    def parse_config(self):
        self.IMAGE_CUTOUT_SIZE = self.config.getint("Handler", "IMAGE_CUTOUT_SIZE")
        self.IMG_ZOOM_CNT = self.config.getint("Handler", "IMG_ZOOM_CNT")
        self.SPEC_ZOOM_CNT = self.config.getint("Handler", "SPEC_ZOOM_CNT")
        self.IMG_SPAT_INDEX_ORDER = self.config.getint("Handler", "IMG_SPAT_INDEX_ORDER")
        self.SPEC_SPAT_INDEX_ORDER = self.config.getint("Handler", "SPEC_SPAT_INDEX_ORDER")
        self.CHUNK_SIZE = self.config.get("Handler", "CHUNK_SIZE")
        self.ORIG_CUBE_NAME = self.config.get("Handler", "ORIG_CUBE_NAME")
        self.DENSE_CUBE_NAME = self.config.get("Handler", "DENSE_CUBE_NAME")
        self.INCLUDE_ADDITIONAL_METADATA = self.config.getboolean("Handler", "INCLUDE_ADDITIONAL_METADATA")
        self.INIT_ARRAY_SIZE = self.config.getint("Handler", "INIT_ARRAY_SIZE")
        self.FITS_MEM_MAP = self.config.getboolean("Handler", "FITS_MEM_MAP")
        self.MPIO = self.config.getboolean("Handler", "MPIO")
        self.PARALLEL_MODE = self.config.get("Handler", "PARALLEL_MODE")
        self.LOG_LEVEL = self.config.get("Handler", "LOG_LEVEL")
        self.COMPRESSION = self.config.get("Writer", "COMPRESSION")
        self.COMPRESSION_OPTS = self.config.get("Writer", "COMPRESSION_OPTS")
        self.FLOAT_COMPRESS = self.config.getboolean("Writer", "FLOAT_COMPRESS")
        self.SHUFFLE = self.config.getboolean("Writer", "SHUFFLE")
        self.IMAGE_PATTERN = self.config.get("Writer", "IMAGE_PATTERN")
        self.SPECTRA_PATTERN = self.config.get("Writer", "SPECTRA_PATTERN")
        self.MAX_CUTOUT_REFS = self.config.getint("Writer", "MAX_CUTOUT_REFS")
        self.LIMIT_IMAGE_COUNT = self.config.getint("Writer", "LIMIT_IMAGE_COUNT")
        self.LIMIT_SPECTRA_COUNT = self.config.getint("Writer", "LIMIT_SPECTRA_COUNT")
        self.BATCH_SIZE = self.config.getint("Writer", "BATCH_SIZE")
        self.POLL_INTERVAL = self.config.getfloat("Writer", "POLL_INTERVAL")
        self.C_BOOSTER = self.config.getboolean("Writer", "C_BOOSTER")
        self.CREATE_REFERENCES = self.config.getboolean("Writer", "CREATE_REFERENCES")
        self.CREATE_DENSE_CUBE = self.config.getboolean("Writer", "CREATE_DENSE_CUBE")
        self.OUTPUT_HEAL_ORDER = self.config.getint("Reader", "OUTPUT_HEAL_ORDER")
        self.APPLY_TRANSMISSION_ONLINE = self.config.get("Processor", "APPLY_TRANSMISSION_ONLINE")
        self.REBIN_MIN = self.config.getfloat("Preprocessing", "REBIN_MIN")
        self.REBIN_MAX = self.config.getfloat("Preprocessing", "REBIN_MAX")
        self.REBIN_SAMPLES = self.config.getint("Preprocessing", "REBIN_SAMPLES")
        self.APPLY_REBIN = self.config.getboolean("Preprocessing", "APPLY_REBIN")
        self.APPLY_TRANSMISSION_CURVE = self.config.getboolean("Preprocessing", "APPLY_TRANSMISSION_CURVE")
