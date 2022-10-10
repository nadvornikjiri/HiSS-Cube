import os
from abc import ABC, abstractmethod
from ast import literal_eval
from pathlib import Path

import numpy as np
import ujson
from tqdm.auto import tqdm

from hisscube.processors.data import float_compress
from hisscube.processors.metadata_strategy import MetadataStrategy, require_zoom_grps
from hisscube.processors.metadata_strategy_dataset import get_data_datasets, get_index_datasets, get_healpix_id, \
    get_dataset_resolution_groups, write_dataset, create_additional_datasets
from hisscube.processors.metadata_strategy_tree import require_spatial_grp
from hisscube.utils.astrometry import get_heal_path_from_coords, get_image_center_coords
from hisscube.utils.config import Config
from hisscube.utils.io import H5Connector, get_image_header_dataset
from hisscube.utils.logging import log_timing, HiSSCubeLogger
from hisscube.utils.nexus import set_nx_data, set_nx_interpretation
from hisscube.utils.photometry import Photometry


class ImageMetadataStrategy(ABC):
    def __init__(self, metadata_strategy: MetadataStrategy, config: Config, photometry: Photometry):
        self.metadata_strategy = metadata_strategy
        self.config = config
        self.photometry = photometry
        self.h5_connector: H5Connector = None
        self.logger = HiSSCubeLogger.logger
        self.img_cnt = 0

    def write_metadata_multiple(self, h5_connector, no_attrs=False, no_datasets=False):
        self._set_connector(h5_connector)
        fits_headers = get_image_header_dataset(h5_connector)
        self.clear_sparse_cube(h5_connector)
        self._write_metadata_from_cache(h5_connector, fits_headers, no_attrs, no_datasets)

    def clear_sparse_cube(self, h5_connector):
        grp_name = self.config.ORIG_CUBE_NAME
        if grp_name in h5_connector.file:
            del h5_connector.file[grp_name]

    def write_metadata(self, h5_connector, fits_path, fits_header, no_attrs=False, no_datasets=False):
        self._set_connector(h5_connector)
        metadata = ujson.loads(fits_header)
        self._write_parsed_metadata(metadata, fits_path, no_attrs, no_datasets)

    def _set_connector(self, h5_connector):
        self.h5_connector = h5_connector

    def _write_metadata_from_cache(self, h5_connector, fits_headers, no_attrs, no_datasets):
        self.img_cnt = 0
        self.h5_connector.fits_total_cnt = 0
        for fits_path, header in tqdm(fits_headers, desc="Writing from image cache", position=0, leave=True):
            self._write_metadata_from_header(h5_connector, fits_path, header, no_attrs, no_datasets)

    @log_timing("process_image_metadata")
    def _write_metadata_from_header(self, h5_connector, fits_path, header, no_attrs, no_datasets):
        self._set_connector(h5_connector)
        fits_path = fits_path.decode('utf-8')
        try:
            self.write_metadata(h5_connector, fits_path, header, no_attrs=no_attrs, no_datasets=no_datasets)
            self.img_cnt += 1
            self.h5_connector.fits_total_cnt += 1
        except RuntimeError as e:
            self.logger.warning(
                "Unable to ingest image %s, message: %s" % (fits_path, str(e)))

    @abstractmethod
    def get_resolution_groups(self, metadata, h5_connector):
        raise NotImplementedError

    @abstractmethod
    def _write_parsed_metadata(self, metadata, fits_path, no_attrs, no_datasets):
        raise NotImplementedError

    @abstractmethod
    def write_datasets(self, res_grp_list, data, file_name, offset):
        raise NotImplementedError


def get_image_time(metadata):
    tai_time = metadata["TAI"]
    return tai_time


class TreeImageStrategy(ImageMetadataStrategy):

    def get_resolution_groups(self, metadata, h5_connector):
        reference_coord = get_image_center_coords(metadata)
        spatial_path = get_heal_path_from_coords(metadata, self.config, ra=reference_coord[0],
                                                 dec=reference_coord[1])
        tai_time = get_image_time(metadata)
        spectral_midpoint = self.photometry.get_image_wl(metadata)
        path = "/".join([spatial_path, str(tai_time), str(spectral_midpoint)])
        spectral_grp = h5_connector.file[path]
        for res_grp in spectral_grp:
            yield spectral_grp[res_grp]

    def write_datasets(self, res_grp_list, data, file_name, offset):
        img_datasets = []
        for group in res_grp_list:
            res_tuple = group.name.split('/')[-1]
            wanted_res = next(img for img in data if str(tuple(img["zoom"])) == res_tuple)  # parsing 2D resolution
            img_data = np.dstack((wanted_res["flux_mean"], wanted_res["flux_sigma"]))
            img_data[img_data == np.inf] = np.nan
            if self.config.FLOAT_COMPRESS:
                img_data = float_compress(img_data)
            ds = group[file_name]
            ds.write_direct(img_data)
            img_datasets.append(ds)
        return img_datasets

    def _write_parsed_metadata(self, metadata, fits_path, no_attrs, no_datasets):
        img_datasets = []
        file_name = os.path.basename(fits_path)
        res_grps = self._create_index_tree(metadata)
        if not no_datasets:
            img_datasets = self._create_datasets(file_name, res_grps)
        if not no_attrs:
            self.metadata_strategy.add_metadata(self.h5_connector, metadata, img_datasets)

    def _create_index_tree(self, metadata):
        """
        Creates the index tree for an image.
        Returns HDF5 group - the one where the image dataset should be placed.
        -------

        """
        cube_grp = self.h5_connector.require_semi_sparse_cube_grp()
        spatial_grp = self._require_spatial_grp_structure(metadata, cube_grp)
        time_grp = self._require_time_grp(metadata, spatial_grp)
        img_spectral_grp = self.require_spectral_grp(metadata, time_grp)
        res_grps = self._require_res_grps(metadata, img_spectral_grp)
        return res_grps

    def _create_datasets(self, file_name, parent_grp_list):
        img_datasets = []
        for group in parent_grp_list:
            chunk_size = None
            if self.config.C_BOOSTER:
                if "image_dataset" in group:
                    raise RuntimeError(
                        "There is already an image dataset %s within this resolution group. Trying to insert image %s." % (
                            list(group["image_dataset"]), file_name))
            elif len(group) > 0:
                raise RuntimeError(
                    "There is already an image dataset %s within this resolution group. Trying to insert image %s." % (
                        list(group), file_name))
            res_tuple = self.h5_connector.get_name(group).split('/')[-1]
            img_data_shape = tuple(reversed(literal_eval(res_tuple))) + (2,)
            if self.config.CHUNK_SIZE:
                chunk_size = literal_eval(self.config.CHUNK_SIZE)
            ds = self.h5_connector.create_image_h5_dataset(group, file_name, img_data_shape, chunk_size)
            self.h5_connector.set_attr(ds, "mime-type", "image")
            self.h5_connector.set_attr(ds, "interpretation", "image")
            img_datasets.append(ds)
        return img_datasets

    def _require_spatial_grp_structure(self, metadata, parent_grp):
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
        image_coords = get_image_center_coords(metadata)

        parent_grp = orig_parent
        for order in range(self.config.IMG_SPAT_INDEX_ORDER):
            parent_grp = require_spatial_grp(self.h5_connector, order, parent_grp, image_coords)
            if order == self.config.IMG_SPAT_INDEX_ORDER - 1:
                return parent_grp

    def _require_time_grp(self, metadata, parent_grp):
        tai_time = get_image_time(metadata)
        grp = self.h5_connector.require_group(parent_grp, str(tai_time))
        self.h5_connector.set_attr(grp, "type", "time")
        return grp

    def require_spectral_grp(self, metadata, parent_grp):
        grp = self.h5_connector.require_group(parent_grp, str(
            self.photometry.get_image_wl(metadata)),
                                              track_order=True)
        self.h5_connector.set_attr(grp, "type", "spectral")
        return grp

    def _require_res_grps(self, metadata, parent_grp):
        res_grp_list = []
        x_lower_res = int(metadata["NAXIS1"])
        y_lower_res = int(metadata["NAXIS2"])
        for res_zoom in range(self.config.IMG_ZOOM_CNT):
            res_grp_name = str((x_lower_res, y_lower_res))
            grp = self.h5_connector.require_group(parent_grp, res_grp_name)
            self.h5_connector.set_attr(grp, "type", "resolution")
            self.h5_connector.set_attr(grp, "res_zoom", res_zoom)
            set_nx_data(grp, self.h5_connector)
            res_grp_list.append(grp)
            x_lower_res = int(x_lower_res / 2)
            y_lower_res = int(y_lower_res / 2)
        return res_grp_list

    def _write_metadata_from_cache(self, h5_connector, fits_headers, no_attrs, no_datasets):
        self.img_cnt = 0
        self.h5_connector.fits_total_cnt = 0
        for fits_path, header in fits_headers:
            self._write_metadata_from_header(h5_connector, fits_path, header, no_attrs, no_datasets)


class DatasetImageStrategy(ImageMetadataStrategy):

    def write_datasets(self, res_grp_list, data, file_name, offset):
        return write_dataset(data, res_grp_list, self.config.FLOAT_COMPRESS, offset)

    def get_resolution_groups(self, metadata, h5_connector):
        yield from get_dataset_resolution_groups(h5_connector, self.config.ORIG_CUBE_NAME, self.config.IMG_ZOOM_CNT,
                                                 "images")

    def _write_metadata_from_cache(self, h5_connector, fits_headers, no_attrs, no_datasets):
        img_count = self.h5_connector.get_image_count()
        img_zoom_groups = require_zoom_grps("images", self.h5_connector, self.config.IMG_ZOOM_CNT)
        if not no_datasets:
            self.create_datasets(img_zoom_groups, img_count)
            super()._write_metadata_from_cache(h5_connector, fits_headers, no_attrs, no_datasets)

        self.sort_indices()

    def _write_parsed_metadata(self, metadata, fits_path, no_attrs, no_datasets):
        img_datasets = get_data_datasets(self.h5_connector, "images", self.config.IMG_ZOOM_CNT,
                                         self.config.ORIG_CUBE_NAME)
        fits_name = Path(fits_path).name
        if not no_attrs:
            self.metadata_strategy.add_metadata(self.h5_connector, metadata, img_datasets, self.img_cnt, fits_name)
        self.add_index_entry(metadata, img_datasets)

    def create_datasets(self, img_zoom_groups, img_count):
        for img_zoom, img_zoom_group in enumerate(img_zoom_groups):
            chunk_size = None
            img_shape = (img_count,
                         int(self.config.IMG_RES_Y / (2 ** img_zoom)),
                         int(self.config.IMG_RES_X / (2 ** img_zoom)))
            if self.config.DATASET_STRATEGY_CHUNKED:
                chunk_size = (1,) + img_shape[1:]
            img_ds = self.h5_connector.create_image_h5_dataset(img_zoom_group, "data", img_shape, chunk_size)
            self.h5_connector.set_attr(img_ds, "mime-type", "image")
            set_nx_interpretation(img_ds, "image", self.h5_connector)
            error_ds = self.h5_connector.create_image_h5_dataset(img_zoom_group, "errors", img_shape, chunk_size)
            self.h5_connector.set_attr(img_ds, "error_ds_ref", error_ds.ref)
            index_dtype = [("spatial", np.int64), ("time", np.float32), ("wl", np.int32), ("ds_slice_idx", np.int64)]
            create_additional_datasets(img_count, img_ds, img_zoom_group, index_dtype, self.h5_connector,
                                       self.config.FITS_IMAGE_MAX_HEADER_SIZE, self.config.FITS_MAX_PATH_SIZE)

    def add_index_entry(self, metadata, img_datasets):
        for img_ds in img_datasets:
            index_ds = self.h5_connector.file[self.h5_connector.get_attr(img_ds, "index_ds_ref")]
            image_center_ra, image_center_dec = get_image_center_coords(metadata)
            healpix_id = get_healpix_id(image_center_ra, image_center_dec, self.config.IMG_SPAT_INDEX_ORDER - 1)
            time = get_image_time(metadata)
            wl = self.photometry.get_image_wl(metadata)
            index_ds[self.img_cnt] = (healpix_id, time, wl, self.img_cnt)

    def sort_indices(self):
        index_datasets = get_index_datasets(self.h5_connector, "images", self.config.IMG_ZOOM_CNT,
                                            self.config.ORIG_CUBE_NAME)
        for index_ds in index_datasets:
            index_ds[:] = np.sort(index_ds[:], order=['spatial', 'time', 'wl'])
