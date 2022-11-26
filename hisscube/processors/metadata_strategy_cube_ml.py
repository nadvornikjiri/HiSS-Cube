import traceback
from abc import ABC, abstractmethod
from bisect import bisect_left, bisect_right
from json import JSONDecodeError

import h5py
import numpy as np
import ujson
from tqdm.auto import tqdm

from hisscube.processors.metadata_strategy import MetadataStrategy
from hisscube.processors.metadata_strategy_dataset import DatasetStrategy, get_cutout_data_datasets, \
    get_cutout_metadata_datasets, get_data_datasets, get_error_datasets, get_index_datasets, get_metadata_datasets
from hisscube.processors.metadata_strategy_tree import TreeStrategy
from hisscube.processors.metadata_strategy_spectrum import get_spectrum_time
from hisscube.utils.astrometry import get_cutout_pixel_coords, NoCoverageFoundError
from hisscube.utils.io import get_spectrum_header_dataset, H5Connector, get_error_ds
from hisscube.utils.io_strategy import get_orig_header
from hisscube.utils.logging import HiSSCubeLogger, wrap_tqdm
from hisscube.utils.nexus import add_nexus_navigation_metadata, set_nx_data, set_nx_interpretation, set_nx_signal
from hisscube.utils.photometry import Photometry


def aggregate_inverse_variance_weighting(arr, axis=0):  # TODO rescale by 1e-17 to make the calculations easier?
    arr = arr.astype('<f8')  # necessary conversion as the numbers are small
    flux = arr[..., 0]
    flux_sigma = arr[..., 1]
    weighted_flux_axis = np.divide(flux, flux_sigma ** 2, np.zeros_like(flux), where=flux_sigma != 0)
    weighted_flux_sigma = np.divide(1, flux_sigma ** 2, np.zeros_like(flux_sigma), where=flux_sigma != 0)
    weighted_mean = np.nansum(weighted_flux_axis, axis=axis) / np.nansum(weighted_flux_sigma, axis=0)
    sigma_sum = np.nansum(weighted_flux_sigma, axis=axis)
    sigma_ratio = np.divide(1, sigma_sum, np.zeros_like(sigma_sum), where=sigma_sum != 0)
    weighted_sigma = np.sqrt(sigma_ratio)
    res = np.stack((weighted_mean, weighted_sigma), axis=-1)
    return res.astype('<f4')


def target_distance(arr1, arr2):
    arr1 = arr1.astype('<f8')
    arr2 = arr2.astype('<f8')
    flux1, flux_sigma1 = arr1[..., 0], arr1[..., 1]
    flux2, flux_sigma2 = arr2[..., 0], arr2[..., 1]
    arr1_weighted = (flux1 / flux_sigma1 ** 2) / (1 / flux_sigma1 ** 2)
    arr2_weighted = (flux2 / flux_sigma2 ** 2) / (1 / flux_sigma2 ** 2)
    diff = np.nansum(np.absolute(arr1_weighted - arr2_weighted))
    return float(diff)


class SparseTreeCube:
    def __init__(self, data=None):
        self.data = data
        self.image_cutout_refs = {}
        self.image_metadata_refs = {}
        self.spec_metadata_refs = {}
        if self.data is None:
            self.data = {}


class MLProcessorStrategy(ABC):

    def __init__(self, config, metadata_strategy: MetadataStrategy, photometry: Photometry):
        self.metadata_strategy = metadata_strategy
        self.img_region_ref_dtype = metadata_strategy.img_region_ref_dtype
        self.config = config
        self.photometry = photometry
        self.logger = HiSSCubeLogger.logger
        self.spectral_3d_cube = None
        self.spec_3d_cube_datasets = {"spectrum": {}, "image": {}, "image_cutout_refs": {}, "image_metadata_refs": {},
                                      "spec_metadata_refs": {}}
        self.current_target_cnt = {}
        self.image_cube_buffer = {}
        self.spec_cube_buffer = {}
        self.current_target_image_buffer = {}
        self.current_target_spec_buffer = {}
        self.image_cutout_refs_buffer = None
        self.image_metadata_refs_buffer = None
        self.spec_metadata_refs_buffer = None
        self.target_cnt = 0

    def clear_buffers(self):
        del self.image_cube_buffer
        del self.spec_cube_buffer
        del self.image_cutout_refs_buffer
        del self.image_metadata_refs_buffer
        del self.spec_metadata_refs_buffer
        del self.current_target_image_buffer
        del self.current_target_spec_buffer
        self.image_cube_buffer = {}
        self.spec_cube_buffer = {}
        self.current_target_image_buffer = {}
        self.current_target_spec_buffer = {}
        self.image_cutout_refs_buffer = None
        self.image_metadata_refs_buffer = None
        self.spec_metadata_refs_buffer = None

    @abstractmethod
    def create_3d_cube(self, h5_connector):
        raise NotImplementedError

    @staticmethod
    def _aggregate_3d_cube(cutout_data, spec_data):
        target_spectra_1d_cube1d_cube = aggregate_inverse_variance_weighting(spec_data)
        target_image_3d_cube = []
        for wl in cutout_data:
            stacked_cutout_for_wl = aggregate_inverse_variance_weighting(cutout_data[wl])
            target_image_3d_cube.append(stacked_cutout_for_wl)

        target_image_3d_cube = np.array(target_image_3d_cube)
        return target_image_3d_cube, target_spectra_1d_cube1d_cube

    def create_datasets_for_zoom(self, h5_connector: H5Connector, cutout_size, dense_grp, target_count, rebin_samples,
                                 zoom, copy=False):
        spectral_dshape = (target_count,
                           int(rebin_samples / 2 ** zoom))
        if self.config.ML_CUBE_CHUNK_SIZE > target_count:
            spectral_stack_chunk = target_count
        else:
            spectral_stack_chunk = self.config.ML_CUBE_CHUNK_SIZE
        spectral_chunk_size = (spectral_stack_chunk,) + spectral_dshape[1:]
        image_dshape = (target_count,
                        5,  # number of image bands that can cover spectrum.
                        int(cutout_size / 2 ** zoom),
                        int(cutout_size / 2 ** zoom))
        image_chunk_size = (spectral_stack_chunk,) + image_dshape[1:]
        dtype = np.dtype('<f4')  # both mean and sigma values are float
        if not copy:
            image_ml_grp, spec_ml_grp = self.recreate_groups(dense_grp, h5_connector, zoom)
        else:
            res_grp = dense_grp[str(zoom)]
            image_ml_grp = res_grp["ml_image"]
            spec_ml_grp = res_grp["ml_spectrum"]
        self.recreate_data_datasets(dtype, h5_connector, image_chunk_size, image_dshape, image_ml_grp, spec_ml_grp,
                                    spectral_chunk_size, spectral_dshape, zoom, copy)
        self.recreate_index_datasets(h5_connector, zoom, image_ml_grp, spec_ml_grp, target_count, copy)
        self.current_target_cnt[zoom] = 0

    def recreate_index_datasets(self, h5_connector: H5Connector, zoom, image_ml_grp, spec_ml_grp, target_count,
                                copy=False):
        self.recreate_index_dataset(h5_connector, image_ml_grp, target_count, zoom, "image_cutout_refs",
                                    wl_count=self.config.FILTER_CNT, copy=copy)
        self.recreate_index_dataset(h5_connector, image_ml_grp, target_count, zoom, "image_metadata_refs",
                                    wl_count=self.config.FILTER_CNT, copy=copy)
        self.recreate_index_dataset(h5_connector, spec_ml_grp, target_count, zoom, "spec_metadata_refs", copy=copy)

    def recreate_index_dataset(self, h5_connector, image_ml_grp, target_count, zoom, ds_name, wl_count=None,
                               copy=False):
        if copy:
            ds_name += "_copy"
        image_index_ds = h5_connector.recreate_regionref_dataset(ds_name, target_count, image_ml_grp,
                                                                 dtype=self.img_region_ref_dtype, wl_count=wl_count)
        if not copy:
            self.spec_3d_cube_datasets[ds_name][zoom] = image_index_ds

    def recreate_data_datasets(self, dtype, h5_connector, image_chunk_size, image_dshape, image_ml_grp, spec_ml_grp,
                               spectral_chunk_size, spectral_dshape, zoom, copy=False):
        if copy:
            ds_name = "cutout_3d_cube_zoom_%d_copy" % zoom
        else:
            ds_name = "cutout_3d_cube_zoom_%d" % zoom
        if ds_name in image_ml_grp:
            del image_ml_grp[ds_name]
        image_ds = h5_connector.create_dataset(image_ml_grp, ds_name, image_dshape, chunk_size=image_chunk_size,
                                               dataset_type=dtype)
        set_nx_interpretation(image_ds, "image", h5_connector)
        if not copy:
            self.spec_3d_cube_datasets["image"][zoom] = image_ds
            set_nx_signal(image_ml_grp, ds_name, h5_connector)
        self._create_error_ds(h5_connector, image_ml_grp, image_ds, image_dshape, dtype, chunk_size=image_chunk_size,
                              copy=copy)
        if copy:
            ds_name = "spectral_1d_cube_zoom_%d_copy" % zoom
        else:
            ds_name = "spectral_1d_cube_zoom_%d" % zoom
        if ds_name in spec_ml_grp:
            del spec_ml_grp[ds_name]
        spec_ds = h5_connector.create_dataset(spec_ml_grp, ds_name, spectral_dshape, chunk_size=spectral_chunk_size,
                                              dataset_type=dtype)
        set_nx_interpretation(spec_ds, "spectrum", h5_connector)
        if not copy:
            self.spec_3d_cube_datasets["spectrum"][zoom] = spec_ds
            set_nx_signal(spec_ml_grp, ds_name, h5_connector)
        self._create_error_ds(h5_connector, spec_ml_grp, spec_ds, spectral_dshape, dtype,
                              chunk_size=spectral_chunk_size, copy=copy)

    def recreate_groups(self, dense_grp, h5_connector, zoom):
        res_grp = dense_grp.require_group(str(zoom))
        if "ml_image" in res_grp:
            del res_grp["ml_image"]
        image_ml_grp = res_grp.require_group("ml_image")
        set_nx_data(image_ml_grp, h5_connector)
        if "ml_spectrum" in res_grp:
            del res_grp["ml_spectrum"]
        spec_ml_grp = res_grp.require_group("ml_spectrum")
        set_nx_data(spec_ml_grp, h5_connector)
        return image_ml_grp, spec_ml_grp

    @staticmethod
    def get_data(h5_connector, zoom, copy=False):
        if copy:
            copy_suffix = "_copy"
        else:
            copy_suffix = ""
        cutout_3d_cube = h5_connector.file[
            "dense_cube/%d/ml_image/cutout_3d_cube_zoom_%d%s" % (zoom, zoom, copy_suffix)]
        cutout_3d_cube_errors = h5_connector.file["dense_cube/%d/ml_image/errors%s" % (zoom, copy_suffix)]
        spec_1d_cube = h5_connector.file[
            "dense_cube/%d/ml_spectrum/spectral_1d_cube_zoom_%d%s" % (zoom, zoom, copy_suffix)]
        spec_1d_cube_errors = h5_connector.file["dense_cube/%d/ml_spectrum/errors%s" % (zoom, copy_suffix)]
        return cutout_3d_cube, cutout_3d_cube_errors, spec_1d_cube, spec_1d_cube_errors

    @staticmethod
    def get_metadata(h5_connector, zoom, copy=False):
        if copy:
            copy_suffix = "_copy"
        else:
            copy_suffix = ""
        image_cutout_refs_ds = h5_connector.file["dense_cube/%d/ml_image/image_cutout_refs%s" % (zoom, copy_suffix)]
        image_metadata_refs_ds = h5_connector.file["dense_cube/%d/ml_image/image_metadata_refs%s" % (zoom, copy_suffix)]
        spec_metadata_refs_ds = h5_connector.file[
            "dense_cube/%d/ml_spectrum/spec_metadata_refs%s" % (zoom, copy_suffix)]
        return image_cutout_refs_ds, image_metadata_refs_ds, spec_metadata_refs_ds

    @staticmethod
    def get_target_count(h5_connector):
        return h5_connector.file["dense_cube"].attrs["target_count"]

    def _count_spatial_groups_with_depth(self, group, target_depth, curr_depth=0):
        my_cnt = 0
        if curr_depth == target_depth and group.attrs["type"] == "spatial":
            return 1  # increase batch_i
        else:
            for child_grp_name in group.keys():
                child_grp = group[child_grp_name]
                if "type" in child_grp.attrs and child_grp.attrs["type"] == "spatial":
                    my_cnt += self._count_spatial_groups_with_depth(child_grp, target_depth, curr_depth + 1)
            return my_cnt

    @staticmethod
    def _create_error_ds(h5_connector, ml_grp, ds, dshape, dtype, chunk_size=None, copy=False):
        error_ds_name = "errors"
        if copy:
            error_ds_name += "_copy"
        if error_ds_name in ml_grp:
            del ml_grp[error_ds_name]
        error_ds = h5_connector.create_dataset(ml_grp, error_ds_name, dshape, chunk_size=chunk_size, dataset_type=dtype)
        ds.attrs["error_ds"] = error_ds.ref
        return error_ds

    def _get_cutout_cube(self, h5_connector, cutout_cube: SparseTreeCube, spectra_cube, zoom, offset=None,
                         batch_i=None,
                         batch_size=None):
        data = self._get_data(cutout_cube, spectra_cube, zoom)
        metadata = self._get_references(cutout_cube, spectra_cube, zoom)
        return data, metadata

    def _get_data(self, cutout_cube, spectra_cube, zoom_idx):
        if cutout_cube and len(cutout_cube.data) == len(self.photometry.filter_midpoints):  # all filters have cutout
            return self._get_cutout_data(cutout_cube, spectra_cube)

    def _write_data_to_datasets(self, h5_connector, offset, target_cnt, zoom_idx):
        if target_cnt > 0:
            image_cube_ds, image_error_ds, spec_cube_ds, spec_error_ds = self.get_data(h5_connector, zoom_idx)
            image_cube_ds[offset:offset + target_cnt] = self.image_cube_buffer[zoom_idx][
                                                        0:target_cnt, ..., 0]  # Writing values
            image_error_ds[offset:offset + target_cnt] = self.image_cube_buffer[zoom_idx][
                                                         0:target_cnt, ..., 1]  # Writing errors
            spec_cube_ds[offset:offset + target_cnt] = self.spec_cube_buffer[zoom_idx][
                                                       0, 0:target_cnt, ..., 0]  # Writing values
            spec_error_ds[offset:offset + target_cnt] = self.spec_cube_buffer[zoom_idx][
                                                        0, 0:target_cnt, ..., 1]  # Writing errors

    def _get_cutout_data(self, cutout_cube, spectra_cube):
        cutout_data = cutout_cube.data
        spec_data = spectra_cube.data
        target_image_3d_cube, target_spectra_1d_cube = self._aggregate_3d_cube(cutout_data, spec_data)
        return target_image_3d_cube, target_spectra_1d_cube

    def _write_current_target_to_buffer(self, zoom_count, cube_data, cube_metadata, batch_size):
        target_cnt = self.target_cnt
        if min(self.current_target_cnt.values()) > 0:
            for zoom_idx in range(zoom_count):
                target_image_3d_cube, target_spectra_1d_cube = cube_data[zoom_idx]
                cutout_refs, image_metadata_refs, spec_metadata_refs = cube_metadata[zoom_idx]
                if zoom_idx not in self.image_cube_buffer:
                    self.image_cube_buffer[zoom_idx] = np.zeros((batch_size,) + target_image_3d_cube.shape,
                                                                target_image_3d_cube.dtype)
                if zoom_idx not in self.spec_cube_buffer:
                    self.spec_cube_buffer[zoom_idx] = np.zeros((batch_size,) + target_spectra_1d_cube.shape,
                                                               target_spectra_1d_cube.dtype)
                if self.image_cutout_refs_buffer is None:
                    self.image_cutout_refs_buffer = self.metadata_strategy.get_cutout_buffer_per_wl(batch_size,
                                                                                                    self.config.FILTER_CNT)
                if self.image_metadata_refs_buffer is None:
                    self.image_metadata_refs_buffer = self.metadata_strategy.get_cutout_buffer_per_wl(batch_size,
                                                                                                      self.config.FILTER_CNT)
                if self.spec_metadata_refs_buffer is None:
                    self.spec_metadata_refs_buffer = self.metadata_strategy.get_cutout_buffer(batch_size)
                self._write_buffers(target_image_3d_cube, target_spectra_1d_cube, image_metadata_refs, cutout_refs,
                                    spec_metadata_refs, target_cnt, zoom_idx)
                self.current_target_cnt[zoom_idx] = 0  # reset target count for current buffer
            self.target_cnt += 1

    def _write_buffers(self, target_image_3d_cube, target_spectra_1d_cube, image_metadata_refs, cutout_refs,
                       spec_metadata_refs, target_cnt, zoom_idx):
        self.image_cube_buffer[zoom_idx][target_cnt] = target_image_3d_cube
        self.spec_cube_buffer[zoom_idx][target_cnt] = target_spectra_1d_cube
        for wl_idx in range(len(cutout_refs)):
            cutout_cnt = len(cutout_refs[wl_idx])
            self.image_cutout_refs_buffer[zoom_idx, target_cnt, wl_idx, 0:cutout_cnt] = cutout_refs[wl_idx]
            self.image_metadata_refs_buffer[zoom_idx, target_cnt, wl_idx, 0:cutout_cnt] = image_metadata_refs[wl_idx]
        self.spec_metadata_refs_buffer[zoom_idx, target_cnt, 0:len(spec_metadata_refs)] = spec_metadata_refs

    def _get_references(self, cutout_cube, spec_cube, zoom_idx):
        if cutout_cube and len(cutout_cube.data) == len(self.photometry.filter_midpoints):
            return self._get_cube_metadata(cutout_cube, spec_cube, zoom_idx)

    def _get_cube_metadata(self, cutout_cube, spec_cube, zoom_idx):
        image_cutouts = []
        image_metadata = []
        cutout_refs = cutout_cube.image_cutout_refs
        image_metadata_refs = cutout_cube.image_metadata_refs
        spec_metadata_refs = spec_cube.spec_metadata_refs
        for i, wl in enumerate(cutout_refs.keys()):
            image_cutouts.append(cutout_refs[wl])
            image_metadata.append(image_metadata_refs[wl])
        self.current_target_cnt[zoom_idx] += 1
        return image_cutouts, image_metadata, spec_metadata_refs

    def _write_references_to_datasets(self, h5_connector, offset, target_cnt, zoom_idx):
        if target_cnt > 0:
            cutout_refs_ds, image_metadata_refs_ds, spec_metadata_refs_ds = self.get_metadata(h5_connector, zoom_idx)
            cutout_refs_ds[offset:offset + target_cnt, ...] = self.image_cutout_refs_buffer[zoom_idx][
                                                              0:target_cnt, ...]
            image_metadata_refs_ds[offset:offset + target_cnt, ...] = self.image_metadata_refs_buffer[zoom_idx][
                                                                      0:target_cnt, ...]
            spec_metadata_refs_ds[offset:offset + target_cnt, ...] = self.spec_metadata_refs_buffer[zoom_idx][
                                                                     0:target_cnt, ...]

    def _get_image_cutouts(self, cutout_wl, image_cutouts, image_region, data_ref=None, metadata_ref=None):
        if image_cutouts is None:
            image_cutouts = SparseTreeCube()
        cutout_data = image_cutouts.data
        cutout_refs = image_cutouts.image_cutout_refs
        metadata_refs = image_cutouts.image_metadata_refs
        if cutout_wl not in cutout_data:
            cutout_data[cutout_wl] = []
            cutout_refs[cutout_wl] = []
            metadata_refs[cutout_wl] = []
        cutout_data[cutout_wl].append(image_region)
        cutout_refs[cutout_wl].append(data_ref)
        metadata_refs[cutout_wl].append(metadata_ref)
        return cutout_data, image_cutouts

    def _construct_cutout_cube(self, sparse_cube, cutout_data=None):
        if sparse_cube:
            if sparse_cube.spec_metadata_refs:
                sparse_cube.spec_metadata_refs = np.array(sparse_cube.spec_metadata_refs,
                                                          dtype=self.img_region_ref_dtype)
            if cutout_data:
                image_cutout_refs = sparse_cube.image_cutout_refs
                image_metadata_refs = sparse_cube.image_metadata_refs
                for wl in cutout_data.keys():
                    cutout_data[wl] = np.array(cutout_data[wl])
                    image_cutout_refs[wl] = np.array(image_cutout_refs[wl])
                    image_metadata_refs[wl] = np.array(image_metadata_refs[wl])
        return sparse_cube

    def process_data(self, h5_connector, spectra_index_spec_ids_orig_zoom, target_spatial_indices, offset=None,
                     max_range=None, batch_size=None):
        raise NotImplementedError

    def get_targets(self, serial_connector):
        raise NotImplementedError

    def get_entry_points(self, h5_connector):
        raise NotImplementedError

    def recreate_datasets(self, serial_connector, dense_grp, target_count):
        raise NotImplementedError

    def shrink_datasets(self, final_zoom, serial_connector, final_target_cnt):
        raise NotImplementedError

    def recreate_copy_datasets(self, serial_connector, dense_grp, target_count):
        raise NotImplementedError

    def copy_slice(self, h5_connector, old_offset, cnt, new_offset):
        raise NotImplementedError

    def merge_datasets(self, h5_connector):
        raise NotImplementedError


class TreeMLProcessorStrategy(MLProcessorStrategy):
    def __init__(self, config, metadata_strategy: TreeStrategy, photometry: Photometry):
        super().__init__(config, metadata_strategy, photometry)
        self.metadata_strategy: TreeStrategy = metadata_strategy

    def _get_spectral_cube(self, spec_datasets):
        spec_datasets_mean_sigma = self._get_mean_sigma(spec_datasets)
        spectra = SparseTreeCube(spec_datasets_mean_sigma)
        spec_ds = spec_datasets[-1]
        return spec_ds, spectra

    @staticmethod
    def _get_spectrum_header(h5_connector, spec_ds):
        return get_orig_header(h5_connector, spec_ds)

    @staticmethod
    def _get_mean_sigma(spec_datasets):
        return np.array(spec_datasets)[..., 1:3]

    def create_3d_cube(self, h5_connector):
        dense_grp = h5_connector.require_dense_group()

        semi_sparse_grp = h5_connector.file[self.config.SPARSE_CUBE_NAME]
        target_count = self._count_spatial_groups_with_depth(semi_sparse_grp,
                                                             self.config.SPEC_SPAT_INDEX_ORDER)
        dense_grp.attrs["target_count"] = target_count

        for zoom in range(min(self.config.IMG_ZOOM_CNT,
                              self.config.SPEC_ZOOM_CNT)):
            self.create_datasets_for_zoom(h5_connector, self.config.IMAGE_CUTOUT_SIZE, dense_grp, target_count,
                                          self.config.REBIN_SAMPLES, zoom)

        self._append_target_3d_cube(h5_connector, semi_sparse_grp)
        add_nexus_navigation_metadata(h5_connector, self.config)

    def _append_target_3d_cube(self, h5_connector, h5_grp, depth=0):
        if isinstance(h5_grp, h5py.Group):
            if "type" in h5_grp.attrs and \
                    h5_grp.attrs["type"] == "spatial" and \
                    depth == self.config.SPEC_SPAT_INDEX_ORDER:
                target_spectra = {}
                for zoom in range(self.config.SPEC_ZOOM_CNT):
                    target_spectra[zoom] = []
                for time_grp in h5_grp.values():
                    if isinstance(time_grp, h5py.Group):
                        for zoom_idx, res_grp in enumerate(time_grp.values()):
                            for spec_ds in res_grp.values():
                                target_spectra[zoom_idx].append(spec_ds)
                self._write_target_3d_cube(h5_connector, target_spectra)

            else:
                for h5_child_grp in h5_grp.values():
                    if "type" in h5_child_grp.attrs and h5_child_grp.attrs["type"] == "spatial":  # only spatial grps
                        self._append_target_3d_cube(h5_connector, h5_child_grp, depth + 1)

    def _write_target_3d_cube(self, h5_connector, spec_ds_dict):
        for zoom, spec_datasets in spec_ds_dict.items():
            cutout_cube, spectra_cube = self._construct_target_dense_cubes(h5_connector, zoom, spec_datasets)
            if cutout_cube:
                cutout_data = cutout_cube.data
                spec_data = spectra_cube.data
                spec_cube_ds = self.spec_3d_cube_datasets["spectrum"][zoom]
                image_cube_ds = self.spec_3d_cube_datasets["image"][zoom]

                target_image_3d_cube, target_spectra_1d_cube = self._aggregate_3d_cube(cutout_data, spec_data)
                image_cube_ds[self.current_target_cnt[zoom]] = target_image_3d_cube[:, :, :, 0]  # Writing values
                image_error_ds_ref = image_cube_ds.attrs["error_ds"]
                image_error_ds = h5_connector.file[image_error_ds_ref]
                image_error_ds[self.current_target_cnt[zoom]] = target_image_3d_cube[:, :, :, 1]  # Writing errors

                spec_cube_ds[self.current_target_cnt[zoom]] = target_spectra_1d_cube[:, 0]  # Writing values
                spec_error_ds_ref = spec_cube_ds.attrs["error_ds"]
                spec_error_ds = h5_connector.file[spec_error_ds_ref]
                spec_error_ds[self.current_target_cnt[zoom]] = target_spectra_1d_cube[:, 1]  # Writing errors

                self.current_target_cnt[zoom] += 1

    def _construct_target_dense_cubes(self, h5_connector, zoom, spec_datasets):
        spec_ds, spectra = self._get_spectral_cube(spec_datasets)
        image_cutouts = None
        cutout_refs = spec_ds.parent.parent.parent["image_cutouts_%d" % zoom]
        image_cutouts = self._get_image_cutout_cube(h5_connector, cutout_refs, image_cutouts, spec_ds, zoom)
        return image_cutouts, spectra

    def _get_image_cutout_cube(self, h5_connector, cutout_refs, image_cutouts, spec_ds, zoom):

        cutout_data = None
        for region_ref in cutout_refs:
            if region_ref:
                try:
                    image_ds = h5_connector.file[region_ref]
                    image_region = image_ds[region_ref]

                    cutout_bounds, time, w, cutout_wl = self.metadata_strategy.get_cutout_bounds_from_spectrum(
                        h5_connector, image_ds,
                        spec_ds,
                        self.config.IMAGE_CUTOUT_SIZE, zoom)
                    cutout_data, image_cutouts = self._get_image_cutouts(cutout_wl, image_cutouts, image_region)
                except ValueError as e:
                    self.logger.error(
                        "Could not process region for %s, message: %s" % (spec_ds.name, str(e)))
                    self.logger.error(traceback.format_exc())
                    raise e
            else:
                break  # necessary because of how null object references are tested in h5py dataset
        return self._construct_cutout_cube(image_cutouts, cutout_data)


class DatasetMLProcessorStrategy(MLProcessorStrategy):

    def __init__(self, config, metadata_strategy: DatasetStrategy, photometry: Photometry):
        super().__init__(config, metadata_strategy, photometry)
        self.photometry = photometry
        self.metadata_strategy: DatasetStrategy = metadata_strategy

    def create_3d_cube(self, h5_connector):
        dense_grp, target_count, target_spatial_indices = self.get_targets(h5_connector)
        dense_grp, spectra_index_spec_ids_orig_zoom = self.get_entry_points(h5_connector)
        if target_count > 0:
            final_zoom = self.recreate_datasets(h5_connector, dense_grp, target_count)

            final_cnt = self.process_data(h5_connector, spectra_index_spec_ids_orig_zoom, target_spatial_indices)
            self.shrink_datasets(final_zoom, h5_connector, final_cnt)

    def get_entry_points(self, h5_connector: H5Connector):
        dense_grp = h5_connector.get_dense_group()
        spectrum_db_index_datasets = get_index_datasets(h5_connector, "spectra", self.config.SPEC_ZOOM_CNT,
                                                        self.config.SPARSE_CUBE_NAME)
        spectrum_db_index_orig_zoom = spectrum_db_index_datasets[0]
        spectra_index_spec_ids_orig_zoom = spectrum_db_index_orig_zoom[:, "ds_slice_idx"]
        return dense_grp, spectra_index_spec_ids_orig_zoom

    def get_targets(self, h5_connector):
        dense_grp = h5_connector.require_dense_group()
        spectrum_db_index_datasets = get_index_datasets(h5_connector, "spectra", self.config.SPEC_ZOOM_CNT,
                                                        self.config.SPARSE_CUBE_NAME)
        spectrum_db_index_orig_zoom = spectrum_db_index_datasets[0]
        spectra_spatial_index_orig_zoom = spectrum_db_index_orig_zoom[:, "spatial"]
        target_spatial_indices = list(self._get_target_spectra_spatial_ranges(spectra_spatial_index_orig_zoom))
        target_count = len(target_spatial_indices)
        return dense_grp, target_count, target_spatial_indices

    def recreate_datasets(self, h5_connector, dense_grp, target_count, copy=False):
        dense_grp.attrs["target_count"] = target_count
        final_zoom = min(self.config.IMG_ZOOM_CNT, self.config.SPEC_ZOOM_CNT)
        for zoom in range(final_zoom):
            self.create_datasets_for_zoom(h5_connector, self.config.IMAGE_CUTOUT_SIZE, dense_grp, target_count,
                                          self.config.REBIN_SAMPLES, zoom, copy)
        add_nexus_navigation_metadata(h5_connector, self.config)
        return final_zoom

    def recreate_copy_datasets(self, serial_connector, dense_grp, target_count):
        return self.recreate_datasets(serial_connector, dense_grp, target_count, copy=True)

    def shrink_datasets(self, final_zoom, h5_connector, final_target_cnt):
        h5_connector.set_target_count(final_target_cnt)
        for zoom in range(final_zoom):
            data_datasets = self.get_data(h5_connector, zoom)
            metadata_datasets = self.get_metadata(h5_connector, zoom)
            for ds in (data_datasets + metadata_datasets):
                ds.resize(final_target_cnt, axis=0)

    def process_data(self, h5_connector, spectra_index_spec_ids_orig_zoom, target_spatial_indices, offset=None,
                     max_range=None, batch_size=None):
        self.target_cnt = 0
        if not offset:
            offset = 0
        if not max_range:
            max_range = len(target_spatial_indices)
        if not batch_size:
            batch_size = len(target_spatial_indices)
        spectra_data = get_data_datasets(h5_connector, "spectra", self.config.SPEC_ZOOM_CNT,
                                         self.config.SPARSE_CUBE_NAME)
        spectra_errors = get_error_datasets(h5_connector, "spectra", self.config.SPEC_ZOOM_CNT,
                                            self.config.SPARSE_CUBE_NAME)
        iterator = wrap_tqdm(target_spatial_indices, self.config.MPIO, "ML cube spectra")
        for zoom_idx in range(min(self.config.SPEC_ZOOM_CNT, self.config.IMG_ZOOM_CNT)):
            self.current_target_cnt[zoom_idx] = 0
        for i, (spatial_index, db_index_from, db_index_to) in enumerate(iterator):
            try:
                spec_ids = []
                for db_index in range(db_index_from, db_index_to):
                    spec_ids.append(spectra_index_spec_ids_orig_zoom[db_index])
                self._append_target_3d_cube(h5_connector, spectra_data, spectra_errors, spec_ids, offset, i, batch_size)
            except ValueError as e:
                self.logger.warning("Unable to create ML cube slice for spectrum %d" % spatial_index)
                if self.config.LOG_LEVEL == "DEBUG":
                    traceback.format_exc()
                    raise e
        self.clear_buffers()
        return self.target_cnt

    def _get_target_spectra_spatial_ranges(self, spectra_spatial_index_orig_zoom):
        spectra_spatial_healpix_ids = np.unique(spectra_spatial_index_orig_zoom)
        for healpix_id in spectra_spatial_healpix_ids:
            yield healpix_id, bisect_left(spectra_spatial_index_orig_zoom, healpix_id), bisect_right(
                spectra_spatial_index_orig_zoom, healpix_id)

    def _append_target_3d_cube(self, h5_connector, spectra_data, spectra_errors, spec_ids, offset=None, batch_i=None,
                               batch_size=None):
        target_spectra_multiple_zoom_stacked = []
        zoom_count = len(spectra_data)
        for zoom in range(zoom_count):
            target_spectra_multiple_zoom_stacked.append([])
            for spec_idx in spec_ids:
                target_spectrum_data = spectra_data[zoom][spec_idx]
                target_spectrum_errors = spectra_errors[zoom][spec_idx]
                target_spectrum_stacked = np.dstack([target_spectrum_data, target_spectrum_errors])
                target_spectra_multiple_zoom_stacked[zoom].append(target_spectrum_stacked)
        cube_data, cube_metadata = self._get_target_3d_cube(h5_connector, target_spectra_multiple_zoom_stacked,
                                                            spec_ids, offset, batch_i,
                                                            batch_size)
        self._write_current_target_to_buffer(zoom_count, cube_data, cube_metadata, batch_size)
        if batch_i == (batch_size - 1):
            for zoom_idx in range(zoom_count):
                self._write_data_to_datasets(h5_connector, offset, self.target_cnt, zoom_idx)
                self._write_references_to_datasets(h5_connector, offset, self.target_cnt, zoom_idx)

    def _get_target_3d_cube(self, h5_connector, target_spectra, spec_ids, offset=None, batch_i=None,
                            batch_size=None):
        spec_metadata_refs = {}
        cube_data = []
        cube_metadata = []
        for zoom, target_spectra in enumerate(target_spectra):
            cutout_cube, spectra_cube = self._construct_target_dense_cubes(h5_connector, spec_metadata_refs,
                                                                           target_spectra, zoom, spec_ids)
            data, metadata = self._get_cutout_cube(h5_connector, cutout_cube, spectra_cube, zoom, offset, batch_i,
                                                   batch_size)
            if data and metadata:
                cube_data.append(data)
                cube_metadata.append(metadata)
        return cube_data, cube_metadata

    def _construct_target_dense_cubes(self, h5_connector, spec_metadata_refs, target_spectra, zoom, spec_ids):
        image_cutouts = None
        spectrum_original_headers = get_spectrum_header_dataset(h5_connector)
        spectrum_zoom_headers = get_metadata_datasets(h5_connector, "spectra", self.config.SPEC_ZOOM_CNT,
                                                      self.config.SPARSE_CUBE_NAME)
        spectra = self._get_spectral_cube(h5_connector, spec_metadata_refs, target_spectra, spectrum_zoom_headers,
                                          spec_ids, zoom)
        cutout_data_refs = get_cutout_data_datasets(h5_connector, self.config.SPEC_ZOOM_CNT,
                                                    self.config.SPARSE_CUBE_NAME)
        cutout_error_refs = get_cutout_data_datasets(h5_connector, self.config.SPEC_ZOOM_CNT,
                                                     self.config.SPARSE_CUBE_NAME)
        cutout_metadata_refs = get_cutout_metadata_datasets(h5_connector, self.config.SPEC_ZOOM_CNT,
                                                            self.config.SPARSE_CUBE_NAME)
        first_spec_idx = spec_ids[0]  # Image cutouts are taken only for first spectrum, all others would have same
        target_cutout_data_refs = cutout_data_refs[zoom][first_spec_idx]
        target_cutout_error_refs = cutout_error_refs[zoom][first_spec_idx]
        target_cutout_metadata_refs = cutout_metadata_refs[zoom][first_spec_idx]
        target_spectrum_metadata = h5_connector.read_serialized_fits_header(spectrum_original_headers,
                                                                            idx=first_spec_idx)
        image_cutouts = self._get_image_cutout_cube(h5_connector, target_cutout_data_refs, target_cutout_error_refs,
                                                    target_cutout_metadata_refs, target_spectrum_metadata,
                                                    image_cutouts, first_spec_idx, zoom)
        return image_cutouts, spectra

    def _get_spectral_cube(self, h5_connector: H5Connector, spec_metadata_refs, target_spectra, spectra_metadata_ds,
                           spec_ids, spec_zoom):
        spec_datasets_mean_sigma = self._get_mean_sigma(np.array(target_spectra))
        spectra = SparseTreeCube(spec_datasets_mean_sigma)
        if spec_zoom not in spec_metadata_refs:
            spec_metadata_refs[spec_zoom] = []
        for spec_idx in spec_ids:
            spec_metadata_ref = h5_connector.get_metadata_ref(spectra_metadata_ds[spec_zoom], spec_idx)
            spec_metadata_refs[spec_zoom].append(spec_metadata_ref)
            spectra.spec_metadata_refs = spec_metadata_refs[spec_zoom]
        return self._construct_cutout_cube(spectra)

    @staticmethod
    def _get_spectrum_header(spec_from_idx, h5_connector, spec_idx, spectrum_original_headers):
        return h5_connector.read_serialized_fits_header(spectrum_original_headers, idx=spec_from_idx + spec_idx)

    @staticmethod
    def _get_mean_sigma(spec_datasets):
        return spec_datasets

    def _get_image_cutout_cube(self, h5_connector, cutout_data_refs, cutout_error_refs,
                               cutout_metadata_refs, spectrum_metadata, image_cutouts,
                               spec_idx, zoom):
        cutout_data = None
        for data_ref, error_ref, metadata_ref in zip(cutout_data_refs, cutout_error_refs, cutout_metadata_refs):
            if any(data_ref) and any(error_ref) and any(metadata_ref):
                try:
                    image_data_region = h5_connector.dereference_region_ref(data_ref)
                    image_error_region = h5_connector.dereference_region_ref(error_ref)
                    image_metadata_region = h5_connector.dereference_region_ref(metadata_ref)
                    image_fits_header = ujson.loads(image_metadata_region["header"])
                    image_region = np.dstack([image_data_region, image_error_region])
                    cutout_bounds, time, w, cutout_wl = self.metadata_strategy.get_cutout_bounds_from_spectrum(
                        image_fits_header, zoom, spectrum_metadata, self.photometry)
                    cutout_data, image_cutouts = self._get_image_cutouts(cutout_wl, image_cutouts, image_region,
                                                                         data_ref, metadata_ref)
                except (ValueError, NoCoverageFoundError) as e:
                    self.logger.error(
                            "Could not process region for spectrum ID %d, image ref: %s, message: %s" % (spec_idx, data_ref, str(e)))
                    self.logger.error(traceback.format_exc())
                    raise e
            else:
                break  # necessary because of how null object references are tested in h5py dataset
        return self._construct_cutout_cube(image_cutouts, cutout_data)

    def copy_slice(self, h5_connector, old_offset, cnt, new_offset):
        for zoom in range(min(self.config.IMG_ZOOM_CNT, self.config.SPEC_ZOOM_CNT)):
            orig_data_datasets = self.get_data(h5_connector, zoom)
            orig_metadata_datasets = self.get_metadata(h5_connector, zoom)
            copy_data_datasets = self.get_data(h5_connector, zoom, copy=True)
            copy_metadata_datasets = self.get_metadata(h5_connector, zoom, copy=True)
            self.__copy_items(orig_data_datasets, copy_data_datasets, old_offset, cnt, new_offset)
            self.__copy_items(orig_metadata_datasets, copy_metadata_datasets, old_offset, cnt, new_offset)

    @staticmethod
    def __copy_items(old_datasets, new_datasets, old_offset, cnt, new_offset):
        for old_ds, new_ds in zip(old_datasets, new_datasets):
            new_ds[new_offset:new_offset + cnt, ...] = old_ds[old_offset:old_offset + cnt, ...]

    def merge_datasets(self, h5_connector):
        for zoom in range(min(self.config.IMG_ZOOM_CNT, self.config.SPEC_ZOOM_CNT)):
            orig_data_datasets = self.get_data(h5_connector, zoom)
            orig_metadata_datasets = self.get_metadata(h5_connector, zoom)
            copy_data_datasets = self.get_data(h5_connector, zoom, copy=True)
            copy_metadata_datasets = self.get_metadata(h5_connector, zoom, copy=True)
            self.__merge_dataset_list(h5_connector, orig_data_datasets, copy_data_datasets)
            self.__merge_dataset_list(h5_connector, orig_metadata_datasets, copy_metadata_datasets)

    @staticmethod
    def __merge_dataset_list(h5_connector, orig_datasets, copy_datasets):
        for orig_ds, copy_ds in zip(orig_datasets, copy_datasets):
            ds_path = orig_ds.name
            copy_ds_path = copy_ds.name
            del h5_connector.file[ds_path]
            h5_connector.file[ds_path] = copy_ds
            del h5_connector.file[copy_ds_path]
