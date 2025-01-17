from abc import ABC, abstractmethod

import h5py

from hisscube.utils.astrometry import get_image_lower_res_wcs
from hisscube.utils.config import Config
from hisscube.utils.io import H5Connector
from hisscube.utils.nexus import set_nx_data, set_nx_signal
import numpy as np


def recreate_header_ds(h5_connector: H5Connector, max_entries, path_size, header_size, grp, ds_name, chunk_size=None):
    if ds_name in grp:
        del grp[ds_name]
    path_dtype = h5py.string_dtype(encoding="utf-8", length=path_size)
    header_dtype = h5py.string_dtype(encoding="utf-8", length=header_size)
    header_ds_dtype = [("path", path_dtype), ("header", header_dtype)]
    if chunk_size > max_entries:
        chunk_size = 1
    header_ds = h5_connector.create_dataset(grp, ds_name, (max_entries,), chunk_size=(chunk_size,),
                                            dataset_type=header_ds_dtype)
    return header_ds, header_ds_dtype

class MetadataStrategy(ABC):
    def __init__(self, config: Config):
        self.config = config
        dataset_path_type = h5py.string_dtype(encoding="utf-8", length=self.config.MAX_DS_PATH_SIZE)
        self.img_region_ref_dtype = [("ds_path", dataset_path_type), ("ds_slice_idx", np.int64),
                                     ("x_min", np.int32), ("x_max", np.int32),
                                     ("y_min", np.int32), ("y_max", np.int32)]

    @abstractmethod
    def add_metadata(self, h5_connector, metadata, datasets, batch_i=None, batch_size=None, offset=None, fits_name=None,
                     metadata_header_buffer=None, metadata_wcs_buffer=None, recalculate_wcs=False):
        raise NotImplementedError

    def clear_sparse_cube(self, h5_connector):
        grp_name = self.config.SPARSE_CUBE_NAME
        if grp_name in h5_connector.file:
            del h5_connector.file[grp_name]

    def get_cutout_buffer(self, batch_size):
        return np.zeros(
            (min(self.config.IMG_ZOOM_CNT, self.config.SPEC_ZOOM_CNT), batch_size,
             self.config.MAX_CUTOUT_REFS), dtype=self.img_region_ref_dtype)

    def get_cutout_buffer_per_wl(self, batch_size, wl_count):
        return np.zeros(
            (min(self.config.IMG_ZOOM_CNT, self.config.SPEC_ZOOM_CNT), batch_size, wl_count,
             self.config.MAX_CUTOUT_REFS), dtype=self.img_region_ref_dtype)


def write_naxis_values(fits_header, ds_shape):
    naxis = len(ds_shape)
    fits_header["NAXIS"] = naxis
    for axis in range(naxis):
        fits_header["NAXIS%d" % (axis + 1)] = ds_shape[naxis - axis - 1]


def get_lower_res_image_metadata(image_fits_header, orig_image_fits_header, res_idx, wcs=False):
    return get_image_lower_res_wcs(orig_image_fits_header, image_fits_header, res_idx, wcs)


def require_zoom_grps(dataset_type, h5_connector, zoom_cnt):
    semi_sparse_grp = h5_connector.require_semi_sparse_cube_grp()
    for zoom in range(zoom_cnt):
        grp_name = str(zoom)
        zoom_grp = h5_connector.require_group(semi_sparse_grp, grp_name)
        img_grp = h5_connector.require_group(zoom_grp, dataset_type)
        set_nx_data(img_grp, h5_connector)
        set_nx_signal(img_grp, "data", h5_connector)
        yield img_grp
