import logging
import os
from math import log

import h5py
import healpy as hp
import numpy as np

from scripts import SDSSCubeHandler as h5
from scripts import astrometry
from scripts.SDSSCubeReader import SDSSCubeReader
from scripts.astrometry import NoCoverageFoundError, get_optimized_wcs


class SDSSCubeWriter(h5.SDSSCubeHandler):

    def __init__(self, h5_file, cube_utils):
        super(SDSSCubeWriter, self).__init__(h5_file, cube_utils)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.COMPRESSION = None
        self.COMPRESSION_OPTS = None
        self.FLOAT_COMPRESS = True
        self.SHUFFLE = False

    def require_raw_cube_grp(self):
        return self.require_group(self.f, self.ORIG_CUBE_NAME)

    def ingest_image(self, image_path):
        self.metadata, self.data = self.cube_utils.get_multiple_resolution_image(image_path, self.IMG_MIN_RES)
        self.file_name = os.path.basename(image_path)
        res_grps = self.create_image_index_tree()
        img_datasets = self.create_img_datasets(res_grps)
        self.add_metadata(img_datasets)
        self.f.flush()
        return img_datasets

    def ingest_spectrum(self, spec_path):
        self.metadata, self.data = self.cube_utils.get_multiple_resolution_spectrum(spec_path, self.SPEC_MIN_RES)
        self.file_name = os.path.basename(spec_path)
        res_grps = self.create_spectrum_index_tree()
        spec_datasets = self.create_spec_datasets(res_grps)
        self.add_metadata(spec_datasets)
        self.f.flush()
        self.add_image_refs_to_spectra(spec_datasets)
        return spec_datasets

    def create_dense_cube(self):
        reader = SDSSCubeReader(self.f, self.cube_utils)
        spectral_cube = reader.get_spectral_cube_from_orig_for_res(0)  # TODO add dynamically all res_zooms that are available
        ds = self.f.require_dataset(self.DENSE_CUBE_NAME, spectral_cube.shape, spectral_cube.dtype,
                                    compression=self.COMPRESSION,
                                    compression_opts=self.COMPRESSION_OPTS,
                                    shuffle=self.SHUFFLE)
        ds.write_direct(spectral_cube)

    def create_image_index_tree(self):
        cube_grp = self.require_raw_cube_grp()
        spatial_grps = self.require_image_spatial_grp_structure(cube_grp)
        time_grp = self.require_image_time_grp(spatial_grps[0])
        self.add_hard_links(spatial_grps[1:], time_grp)
        img_spectral_grp = self.require_image_spectral_grp(time_grp)
        res_grps = self.require_res_grps(img_spectral_grp)
        return res_grps

    def create_spectrum_index_tree(self):
        spec_grp = self.require_raw_cube_grp()
        spatial_grp = self.require_spectrum_spatial_grp_structure(spec_grp)
        time_grp = self.require_spectrum_time_grp(spatial_grp)
        res_grps = self.require_res_grps(time_grp)
        return res_grps

    def require_image_spatial_grp_structure(self, parent_grp):
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
        spectrum_coord = (self.metadata['PLUG_RA'], self.metadata['PLUG_DEC'])
        for order in range(self.SPEC_SPAT_INDEX_ORDER):
            child_grp = self._require_spatial_grp(order, child_grp, spectrum_coord)
        return child_grp

    def _require_spatial_grp(self, order, prev, coord):
        nside = 2 ** order
        healID = hp.ang2pix(nside, coord[0], coord[1], lonlat=True, nest=True)
        grp = self.require_group(prev, str(healID))  # TBD optimize to 8-byte string?
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
        for i, resolution in enumerate(self.data):
            grp = self.require_group(parent_grp, str(self.data[i]["res"]))
            grp.attrs["type"] = "resolution"
            grp.attrs["res_zoom"] = i
            res_grps.append(grp)
        return res_grps

    def create_img_datasets(self, parent_grp_list):
        img_datasets = []
        for group in parent_grp_list:
            res_tuple = group.name.split('/')[-1]
            wanted_res = next(img for img in self.data if str(img["res"]) == res_tuple)  # parsing 2D resolution
            img_data = np.dstack((wanted_res["flux_mean"], wanted_res["flux_sigma"]))
            if self.FLOAT_COMPRESS:
                img_data = self.float_compress(img_data)
            ds = group.require_dataset(self.file_name, img_data.shape, img_data.dtype,
                                       chunks=self.CHUNK_SIZE,
                                       compression=self.COMPRESSION,
                                       compression_opts=self.COMPRESSION_OPTS,
                                       shuffle=self.SHUFFLE)

            ds.write_direct(img_data)
            ds.attrs["mime-type"] = "image"
            img_datasets.append(ds)
        return img_datasets

    def create_spec_datasets(self, parent_grp_list):
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

    def get_image_heal_path(self):
        pixel_IDs = hp.ang2pix(hp.order2nside(np.arange(self.IMG_SPAT_INDEX_ORDER)),
                               self.metadata["PLUG_RA"],
                               self.metadata["PLUG_DEC"],
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
