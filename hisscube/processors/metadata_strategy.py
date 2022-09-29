from abc import ABC, abstractmethod

import h5py

from hisscube.utils.astrometry import get_image_lower_res_wcs
from hisscube.utils.config import Config
from hisscube.utils.nexus import set_nx_data, set_nx_signal


def get_header_ds(max_entries, path_size, header_size, grp, ds_name):
    path_dtype = h5py.string_dtype(encoding="utf-8", length=path_size)
    header_dtype = h5py.string_dtype(encoding="utf-8", length=header_size)
    header_ds_dtype = [("path", path_dtype), ("header", header_dtype)]
    header_ds = grp.create_dataset(ds_name, (max_entries,),
                                   dtype=header_ds_dtype)
    return header_ds, header_ds_dtype


class MetadataStrategy(ABC):
    def __init__(self, config: Config):
        self.config = config

    @abstractmethod
    def add_metadata(self, h5_connector, metadata, datasets, img_cnt=None, fits_name=None):
        raise NotImplementedError


def write_naxis_values(fits_header, ds_shape):
    naxis = len(ds_shape)
    fits_header["NAXIS"] = naxis
    for axis in range(naxis):
        fits_header["NAXIS%d" % (axis + 1)] = ds_shape[axis]


def get_lower_res_image_metadata(image_fits_header, orig_image_fits_header, res_idx):
    return get_image_lower_res_wcs(orig_image_fits_header, image_fits_header, res_idx)


def require_zoom_grps(dataset_type, connector, zoom_cnt):
    semi_sparse_grp = connector.require_semi_sparse_cube_grp()
    for zoom in range(zoom_cnt):
        grp_name = str(zoom)
        zoom_grp = connector.require_group(semi_sparse_grp, grp_name)
        img_grp = connector.require_group(zoom_grp, dataset_type)
        set_nx_data(img_grp, connector)
        set_nx_signal(img_grp, "data", connector)
        yield img_grp


def dereference_region_ref(region_ref, h5_connector):
    image_ds = h5_connector.file[region_ref]
    image_region = image_ds[region_ref][0]  # TODO check why we need the index 0 here.
    return image_region
