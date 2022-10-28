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
from hisscube.utils.logging import log_timing, HiSSCubeLogger, wrap_tqdm
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

    def write_metadata_multiple(self, h5_connector, no_attrs=False, no_datasets=False, range_min=None, range_max=None,
                                batch_size=None):
        self._set_connector(h5_connector)
        fits_headers = get_image_header_dataset(h5_connector)
        if not self.config.MPIO:
            self.metadata_strategy.clear_sparse_cube(h5_connector)
        self.write_metadata_from_cache(h5_connector, fits_headers, no_attrs, no_datasets, range_min, range_max,
                                       batch_size)

    def write_metadata(self, h5_connector, fits_path, fits_header, no_attrs=False, no_datasets=False, offset=0,
                       batch_size=None):
        self._set_connector(h5_connector)
        metadata = ujson.loads(fits_header)
        self._write_parsed_metadata(metadata, fits_path, no_attrs, no_datasets, offset, batch_size)

    def _set_connector(self, h5_connector):
        self.h5_connector = h5_connector

    def write_metadata_from_cache(self, h5_connector, fits_headers, no_attrs, no_datasets, range_min=None,
                                  range_max=None, batch_size=None):
        self.img_cnt = 0
        self.h5_connector.fits_total_cnt = 0
        if not range_min:
            range_min = 0
        if not range_max:
            range_max = len(fits_headers)
        if not batch_size:
            batch_size = len(fits_headers)
        headers_batch = fits_headers[range_min:range_max]
        iterator = wrap_tqdm(headers_batch, self.config.MPIO, self.__class__.__name__)
        for fits_path, header in iterator:
            self._write_metadata_from_header(h5_connector, fits_path, header, no_attrs, no_datasets, range_min,
                                             batch_size)

    @log_timing("process_image_metadata")
    def _write_metadata_from_header(self, h5_connector, fits_path, header, no_attrs, no_datasets, offset=0,
                                    batch_size=None):
        self._set_connector(h5_connector)
        fits_path = fits_path.decode('utf-8')
        try:
            self.write_metadata(h5_connector, fits_path, header, no_attrs, no_datasets, offset, batch_size)
            self.img_cnt += 1
            self.h5_connector.fits_total_cnt += 1
        except RuntimeError as e:
            self.logger.warning(
                "Unable to ingest image %s, message: %s" % (fits_path, str(e)))

    def clear_buffers(self):
        pass

    @abstractmethod
    def get_resolution_groups(self, metadata, h5_connector):
        raise NotImplementedError

    @abstractmethod
    def _write_parsed_metadata(self, metadata, fits_path, no_attrs, no_datasets, offset, batch_size):
        raise NotImplementedError

    @abstractmethod
    def write_datasets(self, res_grp_list, data, file_name, offset, batch_i, batch_size=1):
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

    def write_datasets(self, res_grp_list, data, file_name, offset, batch_i, batch_size=1):
        img_datasets = []
        for group in res_grp_list:
            res_tuple = group.name.split('/')[-1]
            wanted_res = next(img for img in data if str(tuple(img["zoom_idx"])) == res_tuple)  # parsing 2D resolution
            img_data = np.dstack((wanted_res["flux_mean"], wanted_res["flux_sigma"]))
            img_data[img_data == np.inf] = np.nan
            if self.config.FLOAT_COMPRESS:
                img_data = float_compress(img_data)
            ds = group[file_name]
            ds.write_direct(img_data)
            img_datasets.append(ds)
        return img_datasets

    def _write_parsed_metadata(self, metadata, fits_path, no_attrs, no_datasets, offset, batch_size=None):
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
                    raise AssertionError(
                        "There is already an image dataset %s within this resolution group. Trying to insert image %s." % (
                            list(group["image_dataset"]), file_name))
            elif len(group) > 0:
                raise AssertionError(
                    "There is already an image dataset %s within this resolution group. Trying to insert image %s." % (
                        list(group), file_name))
            res_tuple = self.h5_connector.get_name(group).split('/')[-1]
            img_data_shape = tuple(reversed(literal_eval(res_tuple))) + (2,)
            if self.config.IMAGE_CHUNK_SIZE:
                chunk_size = literal_eval(self.config.IMAGE_CHUNK_SIZE)
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

    def write_metadata_from_cache(self, h5_connector, fits_headers, no_attrs, no_datasets, range_min=None,
                                  range_max=None, batch_size=None):
        self.img_cnt = 0
        self.h5_connector.fits_total_cnt = 0
        for fits_path, header in fits_headers:
            self._write_metadata_from_header(h5_connector, fits_path, header, no_attrs, no_datasets)


class DatasetImageStrategy(ImageMetadataStrategy):
    def __init__(self, metadata_strategy: MetadataStrategy, config: Config, photometry: Photometry):
        super().__init__(metadata_strategy, config, photometry)
        self.buffer = {}
        self.metadata_header_buffer = {}
        self.metadata_index_buffer = {}
        self.datasets_created = False
        self.index_dtype = [("spatial", np.int64), ("time", np.float32), ("wl", np.int32), ("ds_slice_idx", np.int64)]

    def clear_buffers(self):
        del self.buffer
        del self.metadata_index_buffer
        del self.metadata_header_buffer
        self.buffer = {}
        self.metadata_header_buffer = {}
        self.metadata_index_buffer = {}

    def write_datasets(self, res_grp_list, data, file_name, offset, batch_i, batch_size=1):
        return write_dataset(data, res_grp_list, self.config.FLOAT_COMPRESS, offset, buffer=self.buffer,
                             batch_size=batch_size, batch_i=batch_i)

    def get_resolution_groups(self, metadata, h5_connector):
        yield from get_dataset_resolution_groups(h5_connector, self.config.ORIG_CUBE_NAME, self.config.IMG_ZOOM_CNT,
                                                 "images")

    def write_metadata_from_cache(self, h5_connector, fits_headers, no_attrs=False, no_datasets=False, range_min=None,
                                  range_max=None, batch_size=None):
        if not self.config.MPIO:
            self.require_datasets(h5_connector)
        super().write_metadata_from_cache(h5_connector, fits_headers, no_attrs, no_datasets, range_min, range_max,
                                          batch_size)
        if not self.config.MPIO:
            self.sort_indices(h5_connector)

    def require_datasets(self, h5_connector):
        self.h5_connector = h5_connector
        img_count = h5_connector.get_image_count()
        img_zoom_groups = require_zoom_grps("images", h5_connector, self.config.IMG_ZOOM_CNT)
        self.create_datasets(img_zoom_groups, img_count)
        self.datasets_created = True

    def _write_parsed_metadata(self, metadata, fits_path, no_attrs, no_datasets, offset, batch_size):
        img_datasets = get_data_datasets(self.h5_connector, "images", self.config.IMG_ZOOM_CNT,
                                         self.config.ORIG_CUBE_NAME)
        fits_name = Path(fits_path).name
        if not no_attrs:
            self.metadata_strategy.add_metadata(self.h5_connector, metadata, img_datasets, self.img_cnt, batch_size,
                                                offset, fits_name, self.metadata_header_buffer)
        self.add_index_entry(metadata, img_datasets, offset, self.img_cnt, batch_size)

    def create_datasets(self, img_zoom_groups, img_count):
        for img_zoom, img_zoom_group in enumerate(img_zoom_groups):
            chunk_size = None
            img_shape = (img_count,
                         int(self.config.IMG_RES_Y / (2 ** img_zoom)),
                         int(self.config.IMG_RES_X / (2 ** img_zoom)))
            if self.config.DATASET_STRATEGY_CHUNKED:
                if self.config.IMAGE_DATA_BATCH_SIZE > img_count:
                    chunk_stack_size = 1
                else:
                    chunk_stack_size = self.config.IMAGE_DATA_BATCH_SIZE
                chunk_size = (chunk_stack_size,) + img_shape[1:]
            img_ds = self.h5_connector.create_image_h5_dataset(img_zoom_group, "data", img_shape, chunk_size)
            self.h5_connector.set_attr(img_ds, "mime-type", "image")
            set_nx_interpretation(img_ds, "image", self.h5_connector)
            error_ds = self.h5_connector.create_image_h5_dataset(img_zoom_group, "errors", img_shape, chunk_size)
            self.h5_connector.set_attr(img_ds, "error_ds_ref", error_ds.ref)
            create_additional_datasets(img_count, img_ds, img_zoom_group, self.index_dtype, self.h5_connector,
                                       self.config.FITS_IMAGE_MAX_HEADER_SIZE, self.config.FITS_MAX_PATH_SIZE,
                                       self.config.METADATA_CHUNK_SIZE)

    def add_index_entry(self, metadata, img_datasets, offset, batch_i, batch_size):
        for zoom_idx, img_ds in enumerate(img_datasets):
            image_center_ra, image_center_dec = get_image_center_coords(metadata)
            healpix_id = get_healpix_id(image_center_ra, image_center_dec, self.config.IMG_SPAT_INDEX_ORDER - 1)
            time = get_image_time(metadata)
            wl = self.photometry.get_image_wl(metadata)
            if zoom_idx not in self.metadata_index_buffer:
                self.metadata_index_buffer[zoom_idx] = np.zeros((batch_size,), self.index_dtype)
            self.metadata_index_buffer[zoom_idx][batch_i] = (healpix_id, time, wl, offset + batch_i)
            if batch_i == (batch_size - 1):
                index_ds = self.h5_connector.file[self.h5_connector.get_attr(img_ds, "index_ds_ref")]
                index_ds.write_direct(self.metadata_index_buffer[zoom_idx], source_sel=np.s_[0:batch_size],
                                      dest_sel=np.s_[offset:offset + batch_i + 1])

    def sort_indices(self, h5_connector):
        index_datasets = get_index_datasets(h5_connector, "images", self.config.IMG_ZOOM_CNT,
                                            self.config.ORIG_CUBE_NAME)
        for index_ds in index_datasets:
            index_ds[:] = np.sort(index_ds[:], order=['spatial', 'time', 'wl'])
