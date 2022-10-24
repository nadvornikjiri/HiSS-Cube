import healpy
import numpy as np

from hisscube.processors.data import float_compress
from hisscube.processors.metadata_strategy import MetadataStrategy, write_naxis_values, get_lower_res_image_metadata, \
    get_header_ds
from hisscube.utils.astrometry import get_optimized_wcs, get_cutout_bounds
from hisscube.utils.io import get_time_from_image, get_image_header_dataset, H5Connector
from hisscube.utils.io_strategy import write_path


class DatasetStrategy(MetadataStrategy):

    def add_metadata(self, h5_connector, metadata, datasets, idx=None, fits_name=None):
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


def get_healpix_id(ra, dec, order):
    pixel_id = healpy.ang2pix(healpy.order2nside(order), ra, dec, nest=True, lonlat=True)
    return pixel_id


def get_dataset_resolution_groups(h5_connector, semi_sparse_group_name, zoom_cnt, data_type):
    for zoom in range(zoom_cnt):
        path = "%s/%d/%s" % (semi_sparse_group_name, zoom, data_type)
        data_group = h5_connector.file[path]
        yield data_group


def write_dataset(data, res_grp_list, should_compress, offset, coordinates=None, buffer=None, batch_size=None,
                  batch_i=None):
    datasets = []
    for zoom_idx, grp in enumerate(res_grp_list):
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
        if zoom_idx not in buffer:
            buffer[zoom_idx] = np.zeros((2,) + (batch_size,) + data_mean.shape, data_mean.dtype)
        buffer[zoom_idx][0, batch_i, ...] = data_mean
        buffer[zoom_idx][1, batch_i, ...] = data_errors
        if batch_i == (batch_size - 1):
            data_ds = grp["data"]
            error_ds = grp["errors"]
            data_ds.write_direct(buffer[zoom_idx][0], source_sel=np.s_[0:batch_size, ...],
                                 dest_sel=np.s_[offset:offset + batch_i + 1, ...])
            error_ds.write_direct(buffer[zoom_idx][1], source_sel=np.s_[0:batch_size, ...],
                                  dest_sel=np.s_[offset:offset + batch_i + 1, ...])
            datasets.append([data_ds, error_ds])
    return datasets


def create_additional_datasets(img_count, img_ds, img_zoom_group, index_dtype, h5_connector: H5Connector, header_size,
                               path_size, chunk_size):
    ds_name = "db_index"
    if ds_name in img_zoom_group:
        del img_zoom_group[ds_name]
    index_ds = h5_connector.create_dataset(img_zoom_group, "db_index", (img_count,), dataset_type=index_dtype)
    metadata_ds, metadata_ds_dtype = get_header_ds(h5_connector, img_count, path_size, header_size, img_zoom_group,
                                                   "metadata", chunk_size)
    h5_connector.set_attr(img_ds, "metadata_ds_ref", metadata_ds.ref)
    h5_connector.set_attr(img_ds, "index_ds_ref", index_ds.ref)
