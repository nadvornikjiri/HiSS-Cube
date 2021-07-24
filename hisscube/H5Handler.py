import configparser
import logging
import pathlib
from ast import literal_eval as make_tuple
from datetime import datetime
from math import log

import h5py
import healpy as hp
import numpy as np
from astropy.time import Time

from hisscube.astrometry import NoCoverageFoundError, get_cutout_bounds, is_cutout_whole
from hisscube import Photometry as cu


class H5Handler(object):
    def __init__(self, h5_file=None):
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
        # utils
        lib_path = pathlib.Path(__file__).parent.absolute()
        cube_utils = self.cube_utils = cu.Photometry("%s/../config/SDSS_Bands" % lib_path,
                                                     "%s/../config/ccd_gain.tsv" % lib_path,
                                                     "%s/../config/ccd_dark_variance.tsv" % lib_path)
        self.ingest_type = None
        self.spectrum_length = None
        self.image_path_list = []
        self.spectra_path_list = []
        #self.logger = logging.getLogger(self.__class__.__name__)
        self.cube_utils = cube_utils
        self.f = h5_file
        self.h5_path = None
        self.file_name = None
        self.fits_path = None
        self.data = None
        self.metadata = None

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
        cutout_bounds = get_cutout_bounds(image_ds, res_idx, self.metadata,
                                          self.config.getint("Handler", "SPECTRAL_CUTOUT_SIZE"))
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
        return self.require_group(self.f, self.config.get("Handler", "ORIG_CUBE_NAME"))

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
        grp.attrs["type"] = "spatial"
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
            for res_zoom in range(self.config.getint("Handler", "IMG_ZOOM_CNT")):
                res_grp_name = str((x_lower_res, y_lower_res))
                grp = self.require_group(parent_grp, res_grp_name)
                grp.attrs["type"] = "resolution"
                grp.attrs["res_zoom"] = res_zoom
                res_grps.append(grp)
                x_lower_res = int(x_lower_res / 2)
                y_lower_res = int(y_lower_res / 2)
        else:
            x_lower_res = int(self.spectrum_length)
            res_zoom = 0
            for res_zoom in range(self.config.getint("Handler", "SPEC_ZOOM_CNT")):
                res_grp_name = str(x_lower_res)
                grp = self.require_group(parent_grp, res_grp_name)
                grp.attrs["type"] = "resolution"
                grp.attrs["res_zoom"] = res_zoom
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
        orig_ds_link = datasets[0].ref
        for res_idx, ds in enumerate(datasets):
            if res_idx > 0:
                ds.attrs["orig_res_link"] = orig_ds_link
                if ds.attrs["mime-type"] == "image":
                    self.write_lower_res_wcs(ds, res_idx)
            else:
                for key, value in dict(self.metadata).items():
                    if key == "COMMENT":
                        continue    # TODO debug thing
                        to_print = 'COMMENT\n--------\n'
                        for item in value:
                            to_print += item + '\n'
                        ds.parent.create_dataset("COMMENT", data=np.string_(to_print), dtype=unicode_dt)
                    elif key == "HISTORY":
                        continue
                        to_print = 'HISTORY\n--------\n'
                        for item in value:
                            to_print += item + '\n'
                        ds.parent.create_dataset("HISTORY", data=np.string_(to_print), dtype=unicode_dt)
                    else:
                        ds.attrs[key] = value
            naxis = len(ds.shape)
            ds.attrs["NAXIS"] = naxis
            for axis in range(naxis):
                ds.attrs["NAXIS%d" % (axis)] = ds.shape[axis]

    def get_heal_path_from_coords(self, ra=None, dec=None, order=None):
        if ra is None and dec is None:
            ra = self.metadata["PLUG_RA"]
            dec = self.metadata["PLUG_DEC"]
        if order is None:
            order = self.config.getint("Handler", "IMG_SPAT_INDEX_ORDER")
        pixel_IDs = hp.ang2pix(hp.order2nside(np.arange(order)),
                               ra,
                               dec,
                               nest=True,
                               lonlat=True)
        heal_path = "/".join(str(pixel_ID) for pixel_ID in pixel_IDs)
        absolute_path = "%s/%s" % (self.config.get("Handler", "ORIG_CUBE_NAME"), heal_path)
        return absolute_path

    def require_group(self, parent_grp, name, track_order=True):
        if not name in parent_grp:
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
