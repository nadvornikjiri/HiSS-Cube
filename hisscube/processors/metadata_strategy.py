from abc import ABC, abstractmethod

import h5py
import healpy
import numpy as np
from astropy_healpix import healpy

from hisscube.processors.data import float_compress
from hisscube.utils.astrometry import get_image_lower_res_wcs, get_optimized_wcs, get_cutout_bounds
from hisscube.utils.config import Config
from hisscube.utils.io import get_orig_header, get_time_from_image, get_image_header_dataset
from hisscube.utils.io_strategy import write_path
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


def write_naxis_values(fits_header, ds_shape):
    naxis = len(ds_shape)
    fits_header["NAXIS"] = naxis
    for axis in range(naxis):
        fits_header["NAXIS%d" % (axis + 1)] = ds_shape[axis]


def get_lower_res_image_metadata(image_fits_header, orig_image_fits_header, res_idx):
    return get_image_lower_res_wcs(orig_image_fits_header, image_fits_header, res_idx)


class TreeStrategy(MetadataStrategy):

    def add_metadata(self, h5_connector, metadata, datasets, img_cnt=None, fits_name=None):
        """
        Adds spectrum_metadata to the HDF5 data sets of the same image or spectrum in multiple resolutions. It also modifies the
        spectrum_metadata for image where needed and adds the COMMENT and HISTORY attributes as datasets for optimization
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
                fits_header = self._get_lower_res_metadata(datasets, ds, h5_connector, fits_header, res_idx)
            ds_shape = h5_connector.get_shape(ds)
            write_naxis_values(fits_header, ds_shape)
            h5_connector.write_serialized_fits_header(ds, fits_header)

    @staticmethod
    def get_cutout_bounds_from_spectrum(h5_connector, image_ds, spectrum_ds, image_cutout_size, res_idx):
        orig_image_header = get_orig_header(h5_connector, image_ds)
        orig_spectrum_header = get_orig_header(h5_connector, spectrum_ds)
        time = get_time_from_image(orig_image_header)
        wl = image_ds.name.split('/')[-3]
        image_fits_header = h5_connector.read_serialized_fits_header(image_ds)
        w = get_optimized_wcs(image_fits_header)
        cutout_bounds = get_cutout_bounds(image_fits_header, res_idx, orig_spectrum_header,
                                          image_cutout_size)
        return cutout_bounds, time, w, wl

    @staticmethod
    def _get_lower_res_metadata(datasets, ds, h5_connector, fits_header, res_idx):
        if h5_connector.get_attr(ds, "mime-type") == "image":
            h5_connector.set_attr_ref(ds, "orig_res_link", datasets[0])
            orig_image_fits_header = h5_connector.read_serialized_fits_header(datasets[0])
            fits_header = get_lower_res_image_metadata(fits_header, orig_image_fits_header, res_idx)
        return fits_header


class DatasetStrategy(MetadataStrategy):

    def add_metadata(self, h5_connector, metadata, datasets, idx=None, fits_name=None):
        """
        Adds spectrum_metadata to the HDF5 data sets of the same image or spectrum in multiple resolutions. It also modifies the
        spectrum_metadata for image where needed and adds the COMMENT and HISTORY attributes as datasets for optimization
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
                fits_header_metadata = self._get_lower_res_metadata(idx, ds, h5_connector, fits_header_metadata,
                                                                    res_idx)
            ds_shape = h5_connector.get_shape(ds)[1:]  # 1st dimension is number of images or spectra
            write_naxis_values(fits_header_metadata, ds_shape)
            metadata_ds_ref = h5_connector.get_attr(ds, "metadata_ds_ref")
            metadata_ds = h5_connector.file[metadata_ds_ref]
            h5_connector.write_serialized_fits_header(metadata_ds, fits_header_metadata, idx=idx)
            write_path(metadata_ds, fits_name, idx)

    def get_cutout_bounds_from_spectrum(self, image_fits_header, res_idx, spectrum_metadata, photometry):
        time = get_time_from_image(image_fits_header)
        wl = photometry.get_image_wl(image_fits_header)
        w = get_optimized_wcs(image_fits_header)
        cutout_bounds = get_cutout_bounds(image_fits_header, res_idx, spectrum_metadata,
                                          self.config.IMAGE_CUTOUT_SIZE)
        return cutout_bounds, time, w, wl

    def require_spatial_grp(self, h5_connector, order, prev, coord):
        pass

    @staticmethod
    def _get_lower_res_metadata(idx, ds, h5_connector, fits_header, res_idx):
        if h5_connector.get_attr(ds, "mime-type") == "image":
            image_header_ds_orig_res = get_image_header_dataset(h5_connector)
            orig_image_fits_header = h5_connector.read_serialized_fits_header(image_header_ds_orig_res, idx=idx)
            fits_header = get_lower_res_image_metadata(fits_header, orig_image_fits_header, res_idx)
        return fits_header


def get_datasets(h5_connector, data_type, dataset_type, zoom_cnt, semi_sparse_cube_name):
    datasets = []
    for zoom in range(zoom_cnt):
        datasets.append(h5_connector.file["%s/%d/%s/%s" % (semi_sparse_cube_name, zoom, data_type, dataset_type)])
    return datasets


def get_cutout_data_datasets(h5_connector, zoom_cnt, semi_sparse_cube_name):
    dataset_type = "image_cutouts_data"
    data_type = "spectra"
    return get_datasets(h5_connector, data_type, dataset_type, zoom_cnt, semi_sparse_cube_name)


def get_cutout_error_datasets(h5_connector, zoom_cnt, semi_sparse_cube_name):
    dataset_type = "image_cutouts_errors"
    data_type = "spectra"
    return get_datasets(h5_connector, data_type, dataset_type, zoom_cnt, semi_sparse_cube_name)


def get_cutout_metadata_datasets(h5_connector, zoom_cnt, semi_sparse_cube_name):
    dataset_type = "image_cutouts_metadata"
    data_type = "spectra"
    return get_datasets(h5_connector, data_type, dataset_type, zoom_cnt, semi_sparse_cube_name)


def get_data_datasets(h5_connector, data_type, zoom_cnt, semi_sparse_cube_name):
    dataset_type = "data"
    return get_datasets(h5_connector, data_type, dataset_type, zoom_cnt, semi_sparse_cube_name)


def get_error_datasets(h5_connector, data_type, zoom_cnt, semi_sparse_cube_name):
    dataset_type = "errors"
    return get_datasets(h5_connector, data_type, dataset_type, zoom_cnt, semi_sparse_cube_name)


def get_wl_datasets(h5_connector, data_type, zoom_cnt, semi_sparse_cube_name):
    dataset_type = "wl"
    return get_datasets(h5_connector, data_type, dataset_type, zoom_cnt, semi_sparse_cube_name)


def get_index_datasets(h5_connector, data_type, zoom_cnt, semi_sparse_cube_name):
    dataset_type = "db_index"
    return get_datasets(h5_connector, data_type, dataset_type, zoom_cnt, semi_sparse_cube_name)


def get_metadata_datasets(h5_connector, data_type, zoom_cnt, semi_sparse_cube_name):
    dataset_type = "metadata"
    return get_datasets(h5_connector, data_type, dataset_type, zoom_cnt, semi_sparse_cube_name)


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


def write_dataset(data, res_grp_list, should_compress, offset, coordinates=None):
    datasets = []
    for zoom_idx, grp in enumerate(res_grp_list):
        data_ds = grp["data"]
        error_ds = grp["errors"]
        wanted_resolution = data[zoom_idx]
        data_mean = wanted_resolution["flux_mean"]
        data_errors = wanted_resolution["flux_sigma"]
        data_mean[data_mean == np.inf] = np.nan
        data_errors[data_errors == np.inf] = np.nan
        if coordinates:
            if offset == 0:
                coordinates_ds = grp["wl"]  # TODO image coordinates
                wl_coordinates = wanted_resolution["wl"]
                if should_compress:
                    wl_coordinates = float_compress(wl_coordinates)
                coordinates_ds.write_direct(wl_coordinates)
        if should_compress:
            data_mean = float_compress(data_mean)
            data_errors = float_compress(data_errors)
        data_ds.write_direct(data_mean, dest_sel=np.s_[offset, ...])
        error_ds.write_direct(data_errors, dest_sel=np.s_[offset, ...])
        datasets.append([data_ds, error_ds])
    return datasets


def create_additional_datasets(img_count, img_ds, img_zoom_group, index_dtype, h5_connector, header_size, path_size):
    index_ds = h5_connector.require_dataset(img_zoom_group, "db_index", (img_count,), index_dtype)
    metadata_ds, metadata_ds_dtype = get_header_ds(img_count, path_size, header_size, img_zoom_group, "metadata")
    h5_connector.set_attr(img_ds, "metadata_ds_ref", metadata_ds.ref)
    h5_connector.set_attr(img_ds, "index_ds_ref", index_ds.ref)


def dereference_region_ref(region_ref, h5_connector):
    image_ds = h5_connector.file[region_ref]
    image_region = image_ds[region_ref][0]  # TODO check why we need the index 0 here.
    return image_region
