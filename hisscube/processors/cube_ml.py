import h5py

import numpy as np

from hisscube.processors.metadata_strategy_spectrum import get_time_from_spectrum

from hisscube.utils.astrometry import get_cutout_pixel_coords, get_cutout_bounds_from_spectrum
from hisscube.utils.io import get_orig_header
from hisscube.utils.logging import HiSSCubeLogger
from hisscube.utils.nexus import add_nexus_navigation_metadata, set_nx_data, set_nx_interpretation, set_nx_signal, \
    set_nx_axes


def aggregate_inverse_variance_weighting(arr, axis=0):  # TODO rescale by 1e-17 to make the calculations easier?
    arr = arr.astype('<f8')  # necessary conversion as the numbers are small
    flux = arr[..., 0]
    flux_sigma = arr[..., 1]
    weighted_mean = np.nansum(flux / flux_sigma ** 2, axis=axis) / \
                    np.nansum(1 / flux_sigma ** 2, axis=0)
    weighed_sigma = np.sqrt(1 /
                            np.nansum(1 / flux_sigma ** 2, axis=axis))
    res = np.stack((weighted_mean, weighed_sigma), axis=-1)
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


def get_spectral_cube(h5_connector, spec_datasets):
    spec_datasets_mean_sigma = np.array(spec_datasets)[..., 1:3]
    for spec_idx, spec_ds in enumerate(spec_datasets):
        spec_header = get_orig_header(h5_connector, spec_ds)
        if spec_idx == 0:
            spec_dims = {"spatial": [spec_header["PLUG_RA"],
                                     spec_header["PLUG_DEC"]],  # spatial is the same for every spectrum
                         "wl": spec_ds[:, 0],  # wl is the same for every spectrum (binned)
                         "time": []}  # time is different for every spectrum
        spec_dims["time"].append(get_time_from_spectrum(spec_header))
    spectra = SparseTreeCube(spec_datasets_mean_sigma, spec_dims)
    return spec_ds, spectra


def aggregate_3d_cube(cutout_data, cutout_dims, spec_data, spec_dims):
    target_spectra_1d_cube1d_cube = aggregate_inverse_variance_weighting(spec_data)
    target_image_3d_cube = []
    for wl in cutout_data:
        stacked_cutout_for_wl = aggregate_inverse_variance_weighting(cutout_data[wl])
        target_image_3d_cube.append(stacked_cutout_for_wl)
    target_image_3d_cube = np.array(target_image_3d_cube)
    spec_dims["time"] = np.mean(
        spec_dims["time"])  # TODO might change time to probability distribution as well?
    for wl in cutout_dims["child_dim"]:
        time_coords = cutout_dims["child_dim"][wl]["child_dim"]["time"]
        cutout_dims["child_dim"][wl]["child_dim"]["time"] = np.mean(
            time_coords)  # TODO might change time to probability distribution as well?
    return target_image_3d_cube, target_spectra_1d_cube1d_cube


class SparseTreeCube:
    def __init__(self, data=None, dims=None):
        self.data = data
        self.dims = dims
        if self.data is None:
            self.data = {}
        if self.dims is None:
            self.dims = {}


class MLProcessor:
    def __init__(self, config):
        self.h5_connector = None
        self.config = config
        self.logger = HiSSCubeLogger.logger
        self.spectral_3d_cube = None
        self.spec_3d_cube_datasets = {"spectral": {}, "image": {}}
        self.target_cnt = {}

    def create_3d_cube(self, h5_connector):
        self.h5_connector = h5_connector
        self.h5_connector.file = h5_connector.file
        dense_grp = self.h5_connector.require_dense_group()

        semi_sparse_grp = self.h5_connector.file[self.config.ORIG_CUBE_NAME]
        target_count = self._count_spatial_groups_with_depth(semi_sparse_grp,
                                                             self.config.SPEC_SPAT_INDEX_ORDER)
        dense_grp.attrs["target_count"] = target_count

        for zoom in range(min(self.config.IMG_ZOOM_CNT,
                              self.config.SPEC_ZOOM_CNT)):
            self.create_datasets_for_zoom(self.config.IMAGE_CUTOUT_SIZE, dense_grp, target_count,
                                          self.config.REBIN_SAMPLES, zoom)

        self._append_target_3d_cube(semi_sparse_grp)
        add_nexus_navigation_metadata(self.h5_connector, self.config)

    def create_datasets_for_zoom(self, cutout_size, dense_grp, target_count, rebin_samples, zoom):
        spectral_dshape = (target_count,
                           int(rebin_samples / 2 ** zoom))
        image_dshape = (target_count,
                        5,  # no image bands that can cover spectrum.
                        int(cutout_size / 2 ** zoom),
                        int(cutout_size / 2 ** zoom))
        dtype = np.dtype('<f4')  # both mean and sigma values are float
        res_grp = dense_grp.require_group(str(zoom))
        image_ml_grp = res_grp.require_group("ml_image")
        set_nx_data(image_ml_grp, self.h5_connector)

        # set_nx_axes(image_ml_grp, "image_time,image_wl,image_spatial_1,image_spatial_2", self.h5_connector)
        spec_ml_grp = res_grp.require_group("ml_spectrum")

        # set_nx_axes(spec_ml_grp, ".,Y,.", self.h5_connector)
        set_nx_data(spec_ml_grp, self.h5_connector)
        ds_name = "cutout_3d_cube_zoom_%d" % zoom

        image_ds = image_ml_grp.require_dataset(ds_name, image_dshape, dtype)
        set_nx_interpretation(image_ds, "image", self.h5_connector)
        self.spec_3d_cube_datasets["image"][zoom] = image_ds
        self._create_error_ds(image_ml_grp, image_ds, image_dshape, dtype)
        set_nx_signal(image_ml_grp, ds_name, self.h5_connector)
        # image_dimensions = {"time": (target_count,),
        #                     "wl": (5,),  # image bands that can cover spectrum.,
        #                     "spatial": (target_count, int(cutout_size / 2 ** zoom), int(cutout_size / 2 ** zoom))
        #                     }  # time is 1D but grouped by wl
        # self.create_dimension_scales(image_ml_grp, zoom, "image", image_dimensions)

        ds_name = "spectral_1d_cube_zoom_%d" % zoom
        spec_ds = spec_ml_grp.require_dataset(ds_name, spectral_dshape, dtype)
        set_nx_interpretation(spec_ds, "spectrum", self.h5_connector)
        self.spec_3d_cube_datasets["spectral"][zoom] = spec_ds
        self._create_error_ds(spec_ml_grp, spec_ds, spectral_dshape, dtype)
        set_nx_signal(spec_ml_grp, ds_name, self.h5_connector)
        # spec_dimensions = {"spatial": (target_count, 2),
        #                    "wl": (target_count, int(rebin_samples / 2 ** zoom)),
        #                    "time": (target_count, 1)}
        # self.create_dimension_scales(spec_ml_grp, zoom, "spectral", spec_dimensions)

        self.target_cnt[zoom] = 0

    def get_spectrum_3d_cube(self, zoom):
        cutout_3d_cube = self.h5_connector.file["dense_cube/%d/ml_image/cutout_3d_cube_zoom_%d" % (zoom, zoom)]
        cutout_3d_cube_errors = self.h5_connector.file["dense_cube/%d/ml_image/errors" % zoom]
        spec_1d_cube = self.h5_connector.file["dense_cube/%d/ml_spectrum/spectral_1d_cube_zoom_%d" % (zoom, zoom)]
        spec_1d_cube_errors = self.h5_connector.file["dense_cube/%d/ml_spectrum/errors" % zoom]
        return cutout_3d_cube, cutout_3d_cube_errors, spec_1d_cube, spec_1d_cube_errors

    def get_target_count(self):
        return self.h5_connector.file["dense_cube"].attrs["target_count"]

    def _create_dimension_scales(self, ml_grp, zoom, dim_type, dim_names):
        dim_ddtype = np.dtype('<f4')
        for dim_idx, dim_item in enumerate(dim_names.items()):
            dim_name, dim_dshape = dim_item
            dim_ds = ml_grp.require_dataset("%s_%s" % (dim_type, dim_name), dim_dshape, dim_ddtype)
            dim_ds.make_scale(dim_name)
            self.spec_3d_cube_datasets[dim_type][zoom].dims[dim_idx].attach_scale(dim_ds)

    def _count_spatial_groups_with_depth(self, group, target_depth, curr_depth=0):
        my_cnt = 0
        if curr_depth == target_depth and group.attrs["type"] == "spatial":
            return 1  # increase cnt
        else:
            for child_grp_name in group.keys():
                child_grp = group[child_grp_name]
                if "type" in child_grp.attrs and child_grp.attrs["type"] == "spatial":
                    my_cnt += self._count_spatial_groups_with_depth(child_grp, target_depth, curr_depth + 1)
            return my_cnt

    def _append_target_3d_cube(self, h5_grp, depth=0):
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
                self._write_target_3d_cube(target_spectra)

            else:
                for h5_child_grp in h5_grp.values():
                    if "type" in h5_child_grp.attrs and h5_child_grp.attrs["type"] == "spatial":  # only spatial grps
                        self._append_target_3d_cube(h5_child_grp, depth + 1)

    def _write_target_3d_cube(self, spec_ds_dict):
        for zoom, spec_datasets in spec_ds_dict.items():
            cutout_cube, spectra_cube = self._construct_target_dense_cubes(zoom, spec_datasets)
            if cutout_cube:
                cutout_data, cutout_dims = cutout_cube.data, cutout_cube.dims
                spec_data, spec_dims = spectra_cube.data, spectra_cube.dims
                spec_cube_ds = self.spec_3d_cube_datasets["spectral"][zoom]
                image_cube_ds = self.spec_3d_cube_datasets["image"][zoom]

                target_image_3d_cube, target_spectra_1d_cube = aggregate_3d_cube(cutout_data, cutout_dims,
                                                                                 spec_data, spec_dims)
                image_cube_ds[self.target_cnt[zoom]] = target_image_3d_cube[:, :, :, 0]  # Writing values
                image_error_ds_ref = image_cube_ds.attrs["error_ds"]
                image_error_ds = self.h5_connector.file[image_error_ds_ref]
                image_error_ds[self.target_cnt[zoom]] = target_image_3d_cube[:, :, :, 1]  # Writing errors

                spec_cube_ds[self.target_cnt[zoom]] = target_spectra_1d_cube[:, 0]  # Writing values
                spec_error_ds_ref = spec_cube_ds.attrs["error_ds"]
                spec_error_ds = self.h5_connector.file[spec_error_ds_ref]
                spec_error_ds[self.target_cnt[zoom]] = target_spectra_1d_cube[:, 1]  # Writing errors

                # self.write_dimensions(spec_cube_ds, image_cube_ds, cutout_dims, spec_dims, zoom)
                self.target_cnt[zoom] += 1

    def _write_dimensions(self, spec_cube_ds, image_cube_ds, cutout_dims, spec_dims, zoom):
        # for dim_idx, spec_dim in enumerate(spec_dims.items()):
        #     dim_name, coords = spec_dim
        #     coords = np.array(coords)
        #     spec_cube_ds.dims[dim_idx][dim_name][self.target_cnt[zoom]] = coords
        spat_coords = cutout_dims["spatial"]
        image_cube_ds.dims[2]["spatial"][self.target_cnt[zoom]] = spat_coords
        wl_coords = np.array(list((cutout_dims["child_dim"].keys()))).astype('i')
        image_cube_ds.dims[1]["wl"][:] = wl_coords
        time_coords = []
        for wl_dim in cutout_dims["child_dim"]:
            time_coords.append(float(cutout_dims["child_dim"][wl_dim]["child_dim"]["time"]))
        image_cube_ds.dims[0]["time"][self.target_cnt[zoom]] = np.array(time_coords)  # time is reduced to 1D

    def _construct_target_dense_cubes(self, zoom, spec_datasets):
        spec_ds, spectra = get_spectral_cube(self.h5_connector, spec_datasets)
        image_cutouts = None
        cutout_refs = spec_ds.parent.parent.parent["image_cutouts_%d" % zoom]
        image_cutouts = self._get_image_cutout_cube(cutout_refs, image_cutouts, spec_ds, zoom)
        return image_cutouts, spectra

    def _get_image_cutout_cube(self, cutout_refs, image_cutouts, spec_ds, zoom):

        for region_ref in cutout_refs:
            if region_ref:
                try:
                    image_ds = self.h5_connector.file[region_ref]
                    image_region = image_ds[region_ref]

                    cutout_bounds, time, w, cutout_wl = get_cutout_bounds_from_spectrum(self.h5_connector, image_ds,
                                                                                        zoom,
                                                                                        spec_ds,
                                                                                        self.config.IMAGE_CUTOUT_SIZE)
                    ra, dec = get_cutout_pixel_coords(cutout_bounds, w)

                    if image_cutouts is None:
                        image_cutouts = SparseTreeCube()
                        cutout_dims = image_cutouts.dims
                        cutout_data = image_cutouts.data
                        cutout_dims["spatial"] = np.stack((ra, dec), axis=2)
                        cutout_dims["child_dim"] = {}

                    wl_dim = cutout_dims["child_dim"]

                    if cutout_wl not in wl_dim:
                        wl_dim[cutout_wl] = {"child_dim": {"time": []}}
                        cutout_data[cutout_wl] = []
                        time_dim = wl_dim[cutout_wl]["child_dim"]["time"]

                    else:
                        time_dim = cutout_dims["child_dim"][cutout_wl]["child_dim"]["time"]
                    cutout_dense_data = cutout_data[cutout_wl]
                    cutout_dense_data.append(image_region)
                    time_dim.append(time)
                except ValueError as e:
                    self.logger.error(
                        "Could not process region for %s, message: %s" % (spec_ds.name, str(e)))
            else:
                break  # necessary because of how null object references are tested in h5py dataset
        if image_cutouts:
            for wl, arr in cutout_data.items():
                cutout_data[wl] = np.array(arr)
            for wl in cutout_dims["child_dim"]:
                cutout_dims["child_dim"][wl]["child_dim"]["time"] = np.array(
                    cutout_dims["child_dim"][wl]["child_dim"]["time"])
        return image_cutouts

    @staticmethod
    def _create_error_ds(ml_grp, ds, dshape, dtype):
        error_ds = ml_grp.require_dataset("errors", dshape, dtype)
        ds.attrs["error_ds"] = error_ds.ref
