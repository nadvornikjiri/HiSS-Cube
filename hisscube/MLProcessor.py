import h5py

from hisscube.H5Handler import H5Handler
import numpy as np

from hisscube.Processor import Processor


class SparseTreeCube():
    def __init__(self, data={}, dims={}):
        self.data = data
        self.dims = dims


class MLProcessor(Processor):
    def __init__(self, h5_file=None):
        super(MLProcessor, self).__init__(h5_file)
        self.spectral_3d_cube = None
        self.spec_3d_cube_datasets = {"spectral": {}, "image": {}}
        self.target_cnt = {}

    def create_3d_cube(self):
        cutout_size = self.config.getint("Handler", "IMAGE_CUTOUT_SIZE")
        rebin_samples = self.config.getint("Preprocessing", "REBIN_SAMPLES")
        dense_grp = self.f[self.config.get("Handler", "DENSE_CUBE_NAME")]
        semi_sparse_grp = self.f[self.config.get("Handler", "ORIG_CUBE_NAME")]
        no_targets = self.count_spatial_groups_with_depth(semi_sparse_grp,
                                                          self.config.getint("Handler", "SPEC_SPAT_INDEX_ORDER"))
        dense_grp.attrs["no_targets"] = no_targets

        for zoom in range(min(self.config.getint("Handler", "IMG_ZOOM_CNT"),
                              self.config.getint("Handler", "SPEC_ZOOM_CNT"))):
            self.create_datasets_for_zoom(cutout_size, dense_grp, no_targets, rebin_samples, zoom)

        self.append_target_3d_cube(semi_sparse_grp)

    def create_datasets_for_zoom(self, cutout_size, dense_grp, no_targets, rebin_samples, zoom):
        spectral_dshape = (no_targets,
                           int(rebin_samples / 2 ** zoom),
                           2)
        image_dshape = (no_targets,
                        5,  # no image bands that can cover spectrum.
                        int(cutout_size / 2 ** zoom),
                        int(cutout_size / 2 ** zoom),
                        2)
        dtype = np.dtype('<f4')  # both mean and sigma values are float
        res_grp = dense_grp[str(zoom)]
        ml_grp = res_grp.require_group("ml")
        self.spec_3d_cube_datasets["spectral"][zoom] = ml_grp.require_dataset("spectral_1d_cube_zoom_%d" % zoom,
                                                                              spectral_dshape, dtype)
        self.spec_3d_cube_datasets["image"][zoom] = ml_grp.require_dataset("cutout_3d_cube_zoom_%d" % zoom,
                                                                           image_dshape, dtype)
        spec_dimensions = {"spatial": (no_targets, 2),
                           "wl": (no_targets, int(rebin_samples / 2 ** zoom)),
                           "time": (no_targets, 1)}
        image_dimensions = {"spatial": (no_targets, int(cutout_size / 2 ** zoom), int(cutout_size / 2 ** zoom), 2),
                            "wl": (no_targets, 5),  # image bands that can cover spectrum.,
                            "time": (no_targets, 5)}  # time is 1D but grouped by wl
        self.create_dimension_scales(ml_grp, zoom, "spectral", spec_dimensions)
        self.create_dimension_scales(ml_grp, zoom, "image", image_dimensions)
        self.target_cnt[zoom] = 0

    def get_spectrum_3d_cube(self, zoom):
        cutout_3d_cube = self.f["dense_cube/%d/ml/cutout_3d_cube_zoom_%d" % (zoom, zoom)]
        spec_1d_cube = self.f["dense_cube/%d/ml/spectral_1d_cube_zoom_%d" % (zoom, zoom)]
        return cutout_3d_cube, spec_1d_cube

    def get_no_targets(self):
        return self.f["dense_cube"].attrs["no_targets"]

    def create_dimension_scales(self, ml_grp, zoom, dim_type, dim_names):
        dim_ddtype = np.dtype('<f4')
        for dim_idx, dim_item in enumerate(dim_names.items()):
            dim_name, dim_dshape = dim_item
            dim_ds = ml_grp.require_dataset("%s_%s" % (dim_type, dim_name), dim_dshape, dim_ddtype)
            dim_ds.make_scale(dim_name)
            self.spec_3d_cube_datasets[dim_type][zoom].dims[dim_idx].attach_scale(dim_ds)

    def count_spatial_groups_with_depth(self, group, target_depth, curr_depth=0):
        my_cnt = 0
        if curr_depth == target_depth and group.attrs["type"] == "spatial":
            return 1  # increase cnt
        else:
            for child_grp_name in group.keys():
                child_grp = group[child_grp_name]
                if "type" in child_grp.attrs and child_grp.attrs["type"] == "spatial":
                    my_cnt += self.count_spatial_groups_with_depth(child_grp, target_depth, curr_depth + 1)
            return my_cnt

    def append_target_3d_cube(self, h5_grp, depth=0):
        if isinstance(h5_grp, h5py.Group):
            if "type" in h5_grp.attrs and \
                    h5_grp.attrs["type"] == "spatial" and \
                    depth == self.config.getint("Handler", "SPEC_SPAT_INDEX_ORDER"):
                target_spectra = {}
                for zoom in range(self.config.getint("Handler", "SPEC_ZOOM_CNT")):
                    target_spectra[zoom] = []
                for time_grp in h5_grp.values():
                    if isinstance(time_grp, h5py.Group):
                        for zoom_idx, res_grp in enumerate(time_grp.values()):
                            for spec_ds in res_grp.values():
                                target_spectra[zoom_idx].append(spec_ds)
                self.write_target_3d_cube(target_spectra)

            else:
                for h5_child_grp in h5_grp.values():
                    if "type" in h5_child_grp.attrs and h5_child_grp.attrs["type"] == "spatial":  # only spatial grps
                        self.append_target_3d_cube(h5_child_grp, depth + 1)

    def write_target_3d_cube(self, spec_ds_dict):
        for zoom, spec_datasets in spec_ds_dict.items():
            cutout_cube, spectra_cube = self.construct_target_dense_cubes(zoom, spec_datasets)
            if cutout_cube:
                cutout_data, cutout_dims = cutout_cube.data, cutout_cube.dims
                spec_data, spec_dims = spectra_cube.data, spectra_cube.dims
                spec_cube_ds = self.spec_3d_cube_datasets["spectral"][zoom]
                image_cube_ds = self.spec_3d_cube_datasets["image"][zoom]

                target_image_3d_cube, target_spectra_1d_cube1d_cube = self.aggregate_3d_cube(cutout_data, cutout_dims,
                                                                                             spec_data, spec_dims)
                spec_cube_ds[self.target_cnt[zoom]] = target_spectra_1d_cube1d_cube
                image_cube_ds[self.target_cnt[zoom]] = target_image_3d_cube
                self.write_dimensions(spec_cube_ds, image_cube_ds, cutout_dims, spec_dims, zoom)
                self.target_cnt[zoom] += 1

    def write_dimensions(self, spec_cube_ds, image_cube_ds, cutout_dims, spec_dims, zoom):
        for dim_idx, spec_dim in enumerate(spec_dims.items()):
            dim_name, coords = spec_dim
            coords = np.array(coords)
            spec_cube_ds.dims[dim_idx][dim_name][self.target_cnt[zoom]] = coords
        spat_coords = cutout_dims["spatial"]
        image_cube_ds.dims[0]["spatial"][self.target_cnt[zoom]] = spat_coords
        wl_coords = np.array(list((cutout_dims["child_dim"].keys()))).astype('i')
        image_cube_ds.dims[1]["wl"][self.target_cnt[zoom]] = wl_coords
        time_coords = []
        for wl_dim in cutout_dims["child_dim"]:
            time_coords.append(float(cutout_dims["child_dim"][wl_dim]["child_dim"]["time"]))
        image_cube_ds.dims[2]["time"][self.target_cnt[zoom]] = np.array(time_coords)  # time is reduced to 1D

    def aggregate_3d_cube(self, cutout_data, cutout_dims, spec_data, spec_dims):
        target_spectra_1d_cube1d_cube = self.aggregate_inverse_variance_weighting(spec_data)
        target_image_3d_cube = []
        for wl in cutout_data:
            stacked_cutout_for_wl = self.aggregate_inverse_variance_weighting(cutout_data[wl])
            target_image_3d_cube.append(stacked_cutout_for_wl)
        target_image_3d_cube = np.array(target_image_3d_cube)
        spec_dims["time"] = np.mean(
            spec_dims["time"])  # TODO might change time to probability distribution as well?
        for wl in cutout_dims["child_dim"]:
            time_coords = cutout_dims["child_dim"][wl]["child_dim"]["time"]
            cutout_dims["child_dim"][wl]["child_dim"]["time"] = np.mean(
                time_coords)  # TODO might change time to probability distribution as well?
        return target_image_3d_cube, target_spectra_1d_cube1d_cube

    def construct_target_dense_cubes(self, zoom, spec_datasets):
        spec_ds, spectra = self.get_spectral_cube(spec_datasets)
        image_cutouts = None
        cutout_refs = spec_ds.parent.parent.parent["image_cutouts_%d" % zoom]
        image_cutouts = self.get_image_cutout_cube(cutout_refs, image_cutouts, spec_ds, zoom)
        return image_cutouts, spectra

    def get_image_cutout_cube(self, cutout_refs, image_cutouts, spec_ds, zoom):

        for region_ref in cutout_refs:
            if region_ref:
                try:
                    image_ds = self.f[region_ref]
                    image_region = image_ds[region_ref]

                    cutout_bounds, time, w, cutout_wl = self.get_cutout_bounds_from_spectrum(image_ds, zoom,
                                                                                             spec_ds)
                    ra, dec = self.get_cutout_pixel_coords(cutout_bounds, w)

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

    def get_spectral_cube(self, spec_datasets):
        spec_datasets_mean_sigma = np.array(spec_datasets)[..., 1:3]
        for spec_idx, spec_ds in enumerate(spec_datasets):
            spec_header = self.get_header(spec_ds)
            if spec_idx == 0:
                spec_dims = {"spatial": [spec_header["PLUG_RA"],
                                         spec_header["PLUG_DEC"]],  # spatial is the same for every spectrum
                             "wl": spec_ds[:, 0],  # wl is the same for every spectrum (binned)
                             "time": []}  # time is different for every spectrum
            spec_dims["time"].append(self.get_time_from_spectrum(spec_header))
        spectra = SparseTreeCube(spec_datasets_mean_sigma, spec_dims)
        return spec_ds, spectra

    @staticmethod
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

    @staticmethod
    def target_distance(arr1, arr2):
        arr1 = arr1.astype('<f8')
        arr2 = arr2.astype('<f8')
        flux1, flux_sigma1 = arr1[..., 0], arr1[..., 1]
        flux2, flux_sigma2 = arr2[..., 0], arr2[..., 1]
        arr1_weighted = (flux1 / flux_sigma1 ** 2) / (1 / flux_sigma1 ** 2)
        arr2_weighted = (flux2 / flux_sigma2 ** 2) / (1 / flux_sigma2 ** 2)
        diff = np.nansum(np.absolute(arr1_weighted - arr2_weighted))
        return float(diff)
