from abc import ABC, abstractmethod

import h5py
import healpy
import numpy as np
import ujson
from astropy_healpix import healpy

from hisscube.processors.data import float_compress
from hisscube.utils.astrometry import get_image_lower_res_wcs, get_image_center_coords
from hisscube.utils.config import Config
from hisscube.utils.io import H5Connector
from hisscube.utils.nexus import set_nx_data, set_nx_signal


class MetadataStrategy(ABC):
    def __init__(self, config: Config):
        self.config = config

    @abstractmethod
    def add_metadata(self, h5_connector, metadata, datasets, img_cnt=None):
        raise NotImplementedError


def require_spatial_grp(h5_connector, order, prev, coord):
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
    healID = healpy.ang2pix(nside, coord[0], coord[1], lonlat=True, nest=True)
    grp = h5_connector.require_group(prev, str(healID))  # TODO optimize to 8-byte string?
    h5_connector.set_attr(grp, "type", "spatial")
    return grp


def write_naxis_values(ds, fits_header, h5_connector, ds_shape):
    naxis = len(ds_shape)
    fits_header["NAXIS"] = naxis
    for axis in range(naxis):
        fits_header["NAXIS%d" % (axis)] = ds_shape[axis]


def get_image_metadata(ds, h5_connector, image_fits_header, orig_image_fits_header, res_idx):
    if h5_connector.get_attr(ds, "mime-type") == "image":
        image_fits_header = get_image_lower_res_wcs(orig_image_fits_header, image_fits_header, res_idx)
    return image_fits_header


def get_lower_res_metadata(datasets, ds, h5_connector, image_fits_header, res_idx):
    h5_connector.set_attr_ref(ds, "orig_res_link", datasets[0])
    orig_image_fits_header = h5_connector.read_serialized_fits_header(datasets[0])
    image_fits_header = get_image_metadata(ds, h5_connector, image_fits_header,
                                           orig_image_fits_header, res_idx)
    return image_fits_header


class TreeStrategy(MetadataStrategy):

    def add_metadata(self, h5_connector, metadata, datasets, img_cnt=None):
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
        fits_header = dict(metadata)
        for res_idx, ds in enumerate(datasets):
            if res_idx > 0:
                fits_header = get_lower_res_metadata(datasets, ds, h5_connector, fits_header, res_idx)
            ds_shape = h5_connector.get_shape(ds)
            write_naxis_values(ds, fits_header, h5_connector, ds_shape)
            h5_connector.write_serialized_fits_header(ds, fits_header)


class DatasetStrategy(MetadataStrategy):

    def add_metadata(self, h5_connector, metadata, datasets, img_cnt=None):
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

        fits_header_metadata = dict(metadata)
        for res_idx, ds in enumerate(datasets):
            if res_idx > 0:
                fits_header_metadata = get_image_metadata(ds, h5_connector, fits_header_metadata,
                                                          fits_header_metadata, res_idx)
            ds_shape = h5_connector.get_shape(ds)[1:]  # 1st dimension is number of images or spectra
            write_naxis_values(ds, fits_header_metadata, h5_connector, ds_shape)
            metadata_ds_ref = h5_connector.get_attr(ds, "metadata_ds_ref")
            metadata_ds = h5_connector.file[metadata_ds_ref]
            metadata_ds[img_cnt] = ujson.dumps(fits_header_metadata)

    def require_spatial_grp(self, h5_connector, order, prev, coord):
        pass


def get_data_datasets(h5_connector, type, zoom_cnt, semi_sparse_cube_name):
    for zoom in range(zoom_cnt):
        dataset = h5_connector.file["%s/%d/%s/data" % (semi_sparse_cube_name, zoom, type)]
        yield dataset


def get_index_datasets(h5_connector, type, zoom_cnt, semi_sparse_cube_name):
    for zoom in range(zoom_cnt):
        dataset = h5_connector.file["%s/%d/%s/db_index" % (semi_sparse_cube_name, zoom, type)]
        yield dataset


def require_zoom_grps(dataset_type, connector, zoom_cnt):
    semi_sparse_grp = connector.require_semi_sparse_cube_grp()
    for zoom in range(zoom_cnt):
        grp_name = str(zoom)
        zoom_grp = connector.require_group(semi_sparse_grp, grp_name)
        img_grp = connector.require_group(zoom_grp, dataset_type)
        set_nx_data(img_grp, connector)
        set_nx_signal(img_grp, "data", connector)
        yield img_grp


def get_healpix_id(ra, dec, order):
    pixel_id = healpy.ang2pix(healpy.order2nside(order), ra, dec, nest=True, lonlat=True)
    return pixel_id


def get_dataset_resolution_groups(h5_connector, semi_sparse_group_name, zoom_cnt, data_type):
    for zoom in range(zoom_cnt):
        path = "%s/%d/%s" % (semi_sparse_group_name, zoom, data_type)
        data_group = h5_connector.file[path]
        yield data_group


def write_dataset(data, res_grp_list, should_compress, offset):
    datasets = []
    for zoom_idx, grp in enumerate(res_grp_list):
        data_ds = grp["data"]
        error_ds = grp["errors"]
        wanted_resolution = data[zoom_idx]
        data_mean = wanted_resolution["flux_mean"]
        data_errors = wanted_resolution["flux_sigma"]
        data_mean[data_mean == np.inf] = np.nan
        data_errors[data_errors == np.inf] = np.nan
        if should_compress:
            data_mean = float_compress(data_mean)
            data_errors = float_compress(data_errors)
        data_ds.write_direct(data_mean, dest_sel=np.s_[offset, ...])
        error_ds.write_direct(data_errors, dest_sel=np.s_[offset, ...])
        datasets.append([data_ds, error_ds])
    return datasets


def create_metadata_index_ds(img_count, img_ds, img_zoom_group, index_dtype, connector, max_header_size):
    index_ds = connector.require_dataset(img_zoom_group, "db_index", (img_count,), index_dtype)
    img_metadata_dtype = h5py.string_dtype(encoding="utf-8", length=max_header_size)
    metadata_ds = connector.require_dataset(img_zoom_group, "metadata", (img_count,),
                                            dtype=img_metadata_dtype)
    connector.set_attr(img_ds, "metadata_ds_ref", metadata_ds.ref)
    connector.set_attr(img_ds, "index_ds_ref", index_ds.ref)
