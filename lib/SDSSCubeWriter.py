import logging
import os
import pathlib
from math import log

import fitsio
import h5py
import healpy as hp
import numpy as np

from lib import SDSSCubeHandler as h5
from lib import astrometry
from lib.SDSSCubeReader import SDSSCubeReader
from lib.astrometry import NoCoverageFoundError, get_optimized_wcs
from ast import literal_eval as make_tuple


class SDSSCubeWriter(h5.SDSSCubeHandler):

    def __init__(self, h5_file=None, cube_utils=None):
        """
        Contains additional constants that are relevant only to writing the HDF5.
        Parameters
        ----------
        h5_file     Opened HDF5 File object
        cube_utils  Object - Initialized cube_utils, containing mainly photometry-related constants needed for preprocessing.
        """
        super(SDSSCubeWriter, self).__init__(h5_file, cube_utils)
        self.ingest_type = None
        self.spectrum_length = None
        self.image_path_list = []
        self.spectra_path_list = []
        self.logger = logging.getLogger(self.__class__.__name__)
        self.COMPRESSION = self.config["Writer"]["COMPRESSION"]
        self.COMPRESSION_OPTS = self.config["Writer"]["COMPRESSION_OPTS"]
        self.FLOAT_COMPRESS = self.config.getboolean("Writer", "FLOAT_COMPRESS")
        self.SHUFFLE = self.config.getboolean("Writer", "SHUFFLE")

    def require_raw_cube_grp(self):
        return self.require_group(self.f, self.ORIG_CUBE_NAME)

    def ingest_image(self, image_path):
        """
        Method that writes an image to the opened HDF5 file (self.f).
        Parameters
        ----------
        image_path  String

        Returns     HDF5 Dataset (already written to the file)
        -------

        """
        self.write_image_metadata(image_path)
        self.metadata, self.data = self.cube_utils.get_multiple_resolution_image(image_path, self.IMG_MIN_RES)
        img_datasets = self.write_img_datasets()
        return img_datasets

    def ingest_spectrum(self, spec_path):
        """
        Method that writes a spectrum to the opened HDF5 file (self.f). Needs to be called after all images are already
        ingested, as it also links the spectra to the images via the Region References.
        Parameters
        ----------
        spec_path   String

        Returns     HDF5 dataset (already written to the file)
        -------

        """
        self.metadata, self.data = self.cube_utils.get_multiple_resolution_spectrum(spec_path, self.SPEC_MIN_RES)
        self.file_name = os.path.basename(spec_path)
        res_grps = self.create_spectrum_index_tree()
        spec_datasets = self.create_spec_datasets(res_grps)
        self.add_metadata(spec_datasets)
        self.f.flush()
        self.add_image_refs_to_spectra(spec_datasets)
        return spec_datasets

    def create_dense_cube(self):
        """
        Creates the dense cube Group and datasets, needs to be called after the the images and spectra were already
        ingested.
        Returns
        -------

        """
        reader = SDSSCubeReader(self.f, self.cube_utils)
        spectral_cube = reader.get_spectral_cube_from_orig_for_res(
            0)  # TODO add dynamically all res_zooms that are available
        ds = self.f.require_dataset(self.DENSE_CUBE_NAME, spectral_cube.shape, spectral_cube.dtype,
                                    compression=self.COMPRESSION,
                                    compression_opts=self.COMPRESSION_OPTS,
                                    shuffle=self.SHUFFLE)
        ds.write_direct(spectral_cube)

    def create_image_index_tree(self):
        """
        Creates the index tree for an image.
        Returns HDF5 group - the one where the image dataset should be placed.
        -------

        """
        cube_grp = self.require_raw_cube_grp()
        spatial_grps = self.require_image_spatial_grp_structure(cube_grp)
        time_grp = self.require_image_time_grp(spatial_grps[0])
        self.add_hard_links(spatial_grps[1:], time_grp)
        img_spectral_grp = self.require_image_spectral_grp(time_grp)
        res_grps = self.require_res_grps(img_spectral_grp)
        return res_grps

    def create_spectrum_index_tree(self):
        """
        Creates the index tree for a spectrum.
        Returns HDF5 group - the one where the spectrum dataset should be placed.
        -------

        """
        spec_grp = self.require_raw_cube_grp()
        spatial_grp = self.require_spectrum_spatial_grp_structure(spec_grp)
        time_grp = self.require_spectrum_time_grp(spatial_grp)
        res_grps = self.require_res_grps(time_grp)
        return res_grps

    def require_image_spatial_grp_structure(self, parent_grp):
        """
        creates the spatial part of index for the image. Returns all of the leaf nodes (resolutions) that we want to
        construct.

        Parameters
        ----------
        parent_grp  HDF5 Group

        Returns     [HDF5 Group]
        -------

        """
        orig_parent = parent_grp
        boundaries = astrometry.get_boundary_coords(self.metadata)
        leaf_grp_set = []
        for coord in boundaries:
            parent_grp = orig_parent
            for order in range(self.IMG_SPAT_INDEX_ORDER):
                parent_grp = self._require_spatial_grp(order, parent_grp, coord)
                if order == self.IMG_SPAT_INDEX_ORDER - 1:
                    # only return each leaf group once.
                    if len(leaf_grp_set) == 0 or \
                            not (any(grp.name == parent_grp.name for grp in leaf_grp_set)):
                        leaf_grp_set.append(parent_grp)
        return leaf_grp_set

    def require_spectrum_spatial_grp_structure(self, child_grp):
        """
        Creates the spatial index part for a spectrum. Takes the root group as parameter.
        Parameters
        ----------
        child_grp   HDF5 group

        Returns     HDF5 group
        -------

        """
        spectrum_coord = (self.metadata['PLUG_RA'], self.metadata['PLUG_DEC'])
        for order in range(self.SPEC_SPAT_INDEX_ORDER):
            child_grp = self._require_spatial_grp(order, child_grp, spectrum_coord)
        return child_grp

    def _require_spatial_grp(self, order, prev, coord):
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

    def require_image_time_grp(self, parent_grp):
        tai_time = self.metadata["TAI"]
        grp = self.require_group(parent_grp, str(tai_time))
        grp.attrs["type"] = "time"
        return grp

    def require_spectrum_time_grp(self, parent_grp):
        try:
            time = self.metadata["TAI"]
        except KeyError:
            time = self.metadata["MJD"]
        grp = self.require_group(parent_grp, str(time))
        grp.attrs["type"] = "time"
        return grp

    @staticmethod
    def add_hard_links(parent_groups, child_groups):
        for parent in parent_groups:
            for child_name in child_groups:
                if not child_name in parent:
                    parent[child_name] = child_groups[child_name]

    def require_image_spectral_grp(self, parent_grp):
        grp = self.require_group(parent_grp, str(self.cube_utils.filter_midpoints[self.metadata["filter"]]))
        grp.attrs["type"] = "spectral"
        return grp

    def require_res_grps(self, parent_grp):
        res_grps = []
        if self.ingest_type == "image":  ##if I'm an image
            min_res = self.IMG_MIN_RES
            x_lower_res = int(self.metadata["NAXIS1"])
            y_lower_res = int(self.metadata["NAXIS2"])
            res_zoom = 0
            while not (x_lower_res < min_res or y_lower_res < min_res):
                res_grp_name = str((y_lower_res, x_lower_res))
                grp = self.require_group(parent_grp, res_grp_name)
                grp.attrs["type"] = "resolution"
                grp.attrs["res_zoom"] = res_zoom
                res_grps.append(grp)
                res_zoom += 1
                x_lower_res = int(x_lower_res / 2)
                y_lower_res = int(y_lower_res / 2)
        else:
            min_res = self.SPEC_MIN_RES
            x_lower_res = int(self.spectrum_length)
            res_zoom = 0
            while not (x_lower_res < min_res):
                res_grp_name = str(x_lower_res)
                grp = self.require_group(parent_grp, res_grp_name)
                grp.attrs["type"] = "resolution"
                grp.attrs["res_zoom"] = res_zoom
                res_grps.append(grp)
                res_zoom += 1
                x_lower_res = int(x_lower_res / 2)
        return res_grps

    def create_img_datasets(self, parent_grp_list):
        img_datasets = []
        for group in parent_grp_list:
            res_tuple = group.name.split('/')[-1]
            img_data_shape = make_tuple(res_tuple) + (2,)
            img_data_dtype = np.dtype('f4')

            ds = group.require_dataset(self.file_name, img_data_shape, img_data_dtype,
                                       chunks=self.CHUNK_SIZE,
                                       compression=self.COMPRESSION,
                                       compression_opts=self.COMPRESSION_OPTS,
                                       shuffle=self.SHUFFLE)
            ds.attrs["mime-type"] = "image"
            img_datasets.append(ds)
        return img_datasets

    def create_spec_datasets(self, parent_grp_list):
        spec_datasets = []
        for group in parent_grp_list:
            res = int(group.name.split('/')[-1])
            spec_data_shape = (res,) + (2,)
            spec_data_dtype = np.dtype('f4')

            ds = group.require_dataset(self.file_name, spec_data_shape, spec_data_dtype,
                                       compression=self.COMPRESSION,
                                       compression_opts=self.COMPRESSION_OPTS,
                                       shuffle=self.SHUFFLE)
            ds.attrs["mime-type"] = "spectrum"
            spec_datasets.append(ds)
            group.require_dataset("image_cutouts", (self.MAX_CUTOUT_REFS,), dtype=h5py.regionref_dtype)
        return spec_datasets

    def create_spec_datasets_old(self, parent_grp_list):
        """
        Creates spectral datasets in a given list of groups (individual resolutions). Optionally uses compression,
        but not chunking.
        Parameters
        ----------
        parent_grp_list [HDF5 Group]

        Returns
        -------

        """
        spec_datasets = []
        for group in parent_grp_list:
            res = group.name.split('/')[-1]
            wanted_res = next(spec for spec in self.data if str(spec["res"]) == res)
            spec_data = np.dstack((wanted_res["wl"], wanted_res["flux_mean"], wanted_res["flux_sigma"]))
            if self.COMPRESSION:
                spec_data = self.float_compress(spec_data)
            ds = group.require_dataset(self.file_name, spec_data.shape, spec_data.dtype,
                                       compression=self.COMPRESSION,
                                       compression_opts=self.COMPRESSION_OPTS,
                                       shuffle=self.SHUFFLE)
            ds.write_direct(spec_data)
            ds.attrs["mime-type"] = "spectrum"
            spec_datasets.append(ds)
        return spec_datasets

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
                        to_print = 'COMMENT\n--------\n'
                        for item in value:
                            to_print += item + '\n'
                        ds.parent.create_dataset("COMMENT", data=np.string_(to_print), dtype=unicode_dt)
                    elif key == "HISTORY":
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

    def write_lower_res_wcs(self, ds, res_idx=0):
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
        image_fits_header = ds.attrs
        image_fits_header["CRPIX1"], image_fits_header["CRPIX2"] = w.wcs.crpix
        [[image_fits_header["CD1_1"], image_fits_header["CD1_2"]],
         [image_fits_header["CD2_1"], image_fits_header["CD2_2"]]] = w.wcs.cd
        image_fits_header["CRVAL1"], image_fits_header["CRVAL2"] = w.wcs.crval
        image_fits_header["CTYPE1"], image_fits_header["CTYPE2"] = w.wcs.ctype

    def add_image_refs_to_spectra(self, spec_datasets):
        """
        Adds HDF5 Region references of image cut-outs to spectra attribute "image_cutouts". Throws NoCoverageFoundError
        if the cut-out does not span the whole cutout size for any reason.
        Parameters
        ----------
        spec_datasets   [HDF5 Datasets]

        Returns         [HDF5 Datasets]
        -------

        """
        image_refs = {}
        image_min_res_idx = 0
        for image_res_idx, image_ds in self.find_images_overlapping_spectrum():
            if not image_res_idx in image_refs:
                image_refs[image_res_idx] = []
            try:
                image_refs[image_res_idx].append(self.get_region_ref(image_res_idx, image_ds))
                if image_res_idx > image_min_res_idx:
                    image_min_res_idx = image_res_idx
            except NoCoverageFoundError:
                self.logger.debug("No coverage found for spectrum %s and image %s" % (self.file_name, image_ds))
                pass

        for res in image_refs:
            image_refs[res] = np.array(image_refs[res],
                                       dtype=h5py.regionref_dtype)
        for spec_res_idx, spec_ds in enumerate(spec_datasets):
            if len(image_refs) > 0:
                if spec_res_idx > image_min_res_idx:
                    spec_ds.attrs["image_cutouts"] = image_refs[image_min_res_idx]
                else:
                    spec_ds.attrs["image_cutouts"] = image_refs[spec_res_idx]
            else:
                spec_ds.attrs["image_cutouts"] = []
        return spec_datasets

    def find_images_overlapping_spectrum(self):
        """Finds images in the HDF5 index structure that overlap the spectrum coordinate. Does so by constructing the
        whole heal_path string to the image and it to get the correct Group containing those images. Yields resolution
        index and the image dataset.

        Yields         (int, HDF5 dataset)
        -------
        """
        heal_path = self.get_image_heal_path()
        heal_path_group = self.f[heal_path]
        for time in heal_path_group:
            time_grp = heal_path_group[time]
            for band in time_grp:
                band_grp = time_grp[band]
                if band_grp.attrs["type"] == "spectral":
                    for res_idx, res in enumerate(band_grp):
                        res_grp = band_grp[res]
                        for image in res_grp:
                            image_ds = res_grp[image]
                            try:
                                if image_ds.attrs["mime-type"] == "image":
                                    yield res_idx, image_ds
                            except KeyError:
                                pass

    def get_image_heal_path(self, ra=None, dec=None):
        if ra is None and dec is None:
            ra = self.metadata["PLUG_RA"]
            dec = self.metadata["PLUG_DEC"]
        pixel_IDs = hp.ang2pix(hp.order2nside(np.arange(self.IMG_SPAT_INDEX_ORDER)),
                               ra,
                               dec,
                               nest=True,
                               lonlat=True)
        heal_path = "/".join(str(pixel_ID) for pixel_ID in pixel_IDs)
        absolute_path = "%s/%s" % (self.ORIG_CUBE_NAME, heal_path)
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

    def close_h5_file(self):
        self.f.close()

    def write_images_metadata(self, image_folder):
        for fits_path in pathlib.Path(image_folder).rglob(self.IMAGE_PATTERN):
            self.write_image_metadata(fits_path)

    def write_image_metadata(self, fits_path):
        self.ingest_type = "image"
        self.image_path_list.append(fits_path)
        self.metadata = fitsio.read_header(fits_path)
        self.file_name = os.path.basename(fits_path)
        res_grps = self.create_image_index_tree()
        img_datasets = self.create_img_datasets(res_grps)
        self.add_metadata(img_datasets)

    def write_spectra_metadata(self, spectra_folder):
        for fits_path in pathlib.Path(spectra_folder).rglob(self.SPECTRA_PATTERN):
            self.write_spectrum_metadata(fits_path)

    def write_spectrum_metadata(self, fits_path):
        self.ingest_type = "spectrum"
        self.spectra_path_list.append(fits_path)
        self.metadata = fitsio.read_header(fits_path)
        self.spectrum_length = fitsio.read_header(fits_path, 1)["NAXIS2"]
        self.file_name = os.path.basename(fits_path)
        res_grps = self.create_spectrum_index_tree()
        spec_datasets = self.create_spec_datasets(res_grps)
        self.add_metadata(spec_datasets)

    def ingest_metadata(self, image_path, spectra_path):
        self.write_images_metadata(image_path)
        self.write_spectra_metadata(spectra_path)

    def write_img_datasets(self):
        res_grp_list = self.get_image_resolution_groups()
        img_datasets = []
        for group in res_grp_list:
            res_tuple = group.name.split('/')[-1]
            wanted_res = next(img for img in self.data if str(img["res"]) == res_tuple)  # parsing 2D resolution
            img_data = np.dstack((wanted_res["flux_mean"], wanted_res["flux_sigma"]))
            if self.FLOAT_COMPRESS:
                img_data = self.float_compress(img_data)
            ds = group[self.file_name]
            ds.write_direct(img_data)
            img_datasets.append(ds)
        return img_datasets

    def get_image_resolution_groups(self):
        spatial_path = self.get_image_heal_path(ra=self.metadata["CRVAL1"], dec=self.metadata["CRVAL2"])
        tai_time = self.metadata["TAI"]
        spectral_midpoint = self.cube_utils.filter_midpoints[self.metadata["filter"]]
        path = "/".join([spatial_path, str(tai_time), str(spectral_midpoint)])
        spectral_grp = self.f[path]
        for res_grp in spectral_grp:
            yield spectral_grp[res_grp]

