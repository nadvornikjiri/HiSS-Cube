import numpy as np
import h5py
import healpy as hp
import numpy as np
from scripts import photUtils
from scripts import cubeUtils
import os


class SDSSCubeWriter:

    def __init__(self, h5path, fits_path, filter_curve_path, ccd_gain_path, ccd_dark_var_path):
        self.cube_utils = cubeUtils.CubeUtils(filter_curve_path, ccd_gain_path, ccd_dark_var_path)
        self.IMG_MIN_RES = 128
        self.SPEC_MIN_RES = 256
        self.IMG_SPAT_INDEX_ORDER = 7
        self.SPEC_SPAT_INDEX_ORDER = 13
        self.CHUNK_SIZE = (128, 128, 2) # 128x128 pixels x (mean, var) tuples
        self.FILE = h5path
        self.ORIG_CUBE_NAME = "orig_data_cube"
        self.NO_IMG_RESOLUTIONS = 5
        self.f = h5py.File(self.FILE, 'r+')
        self.fits_path = fits_path
        self.file_name = os.path.basename(fits_path)
        self.data = None
        self.metadata = None

    def require_raw_cube_grp(self):
        return self.f.require_group(self.ORIG_CUBE_NAME)

    def close_hdf5(self):
        self.f.close()

    def ingest_image(self):
        self.metadata, self.data = self.cube_utils.get_multiple_resolution_image(self.fits_path, self.IMG_MIN_RES)
        res_grps = self.create_image_index_tree()
        img_datasets = self.create_img_datasets(res_grps)
        self.add_metadata(img_datasets)
        self.f.flush()
        return img_datasets

    def ingest_spectrum(self):
        self.metadata, self.data = self.cube_utils.get_multiple_resolution_spectrum(self.fits_path, self.SPEC_MIN_RES)
        res_grps = self.create_spectrum_index_tree()
        spec_datasets = self.create_spec_datasets(res_grps)
        self.add_metadata(spec_datasets)
        self.f.flush()
        #self.add_spec_refs(spec_datasets)
        return spec_datasets

    def create_image_index_tree(self):
        cube_grp = self.require_raw_cube_grp()
        spatial_grps = self.require_image_spatial_grp_structure(cube_grp)
        time_grp = self.require_image_time_grp(spatial_grps[0])
        self.add_hard_links(spatial_grps[1:], time_grp)
        img_spectral_grp = self.create_image_spectral_grp(time_grp)
        res_grps = self.require_res_grps(img_spectral_grp)
        return res_grps

    def create_spectrum_index_tree(self):
        spec_grp = self.require_raw_cube_grp()
        spatial_grp = self.require_spectrum_spatial_grp_structure(spec_grp)
        time_grp = self.require_spectrum_time_grp(spatial_grp)
        res_grps = self.require_res_grps(time_grp)
        return res_grps

    def require_image_spatial_grp_structure(self, parent_grp):
        orig_parent = parent_grp
        boundaries = photUtils.get_boundary_coords(self.metadata)
        leaf_grp_set = []
        for coord in boundaries:
            parent_grp = orig_parent
            for order in range(self.IMG_SPAT_INDEX_ORDER):
                parent_grp = self._require_spatial_grp(order, parent_grp, coord)
                if order == self.IMG_SPAT_INDEX_ORDER - 1:
                    # only return each leaf group once.
                    if len(leaf_grp_set) == 0 or \
                            not (next(grp for grp in leaf_grp_set if grp.name == parent_grp.name)):
                        leaf_grp_set.append(parent_grp)
        return leaf_grp_set

    def require_spectrum_spatial_grp_structure(self, child_grp):
        spectrum_coord = (self.metadata['PLUG_RA'], [self.metadata['PLUG_DEC']])
        for order in range(self.SPEC_SPAT_INDEX_ORDER):
            child_grp = self._require_spatial_grp(order, child_grp, spectrum_coord)
        return child_grp

    def _require_spatial_grp(self, order, prev, coord):
        nside = 2 ** order
        healID = hp.ang2pix(nside, coord[0], coord[1], lonlat=True, nest=True)
        return prev.require_group(str(healID))  #TBD optimize to 8-byte string?

    def require_image_time_grp(self, parent_grp):
        return parent_grp.require_group(str(self.metadata["TAI"]))

    def require_spectrum_time_grp(self, parent_grp):
        return parent_grp.require_group(str(self.metadata["MJD"]))

    def add_hard_links(self, parent_groups, child_groups):
        for parent in parent_groups:
            for child in child_groups:
                parent[child.name] = child

    def create_image_spectral_grp(self, parent_grp):
        return parent_grp.require_group(self.metadata["filter"])

    def require_res_grps(self, parent_grp):
        res_grps = []
        for i, resolution in enumerate(self.data):
            res_grps.append(parent_grp.require_group(str(self.data[i]["res"])))
        return res_grps

    def create_img_datasets(self, parent_grp_list):
        img_datasets = []
        for group in parent_grp_list:
            res_tuple = group.name.split('/')[-1]
            wanted_res = next(img for img in self.data if str(img["res"]) == res_tuple) #parsing 2D resolution
            img_data = np.dstack((wanted_res["flux_mean"], wanted_res["flux_var"]))
            ds = group.create_dataset(self.file_name, img_data.shape, chunks=self.CHUNK_SIZE, compression="gzip")
            ds.write_direct(img_data)
            img_datasets.append(ds)
        return img_datasets

    def create_spec_datasets(self, parent_grp_list):
        spec_datasets = []
        for group in parent_grp_list:
            res = group.name.split('/')[-1]
            wanted_res = next(spec for spec in self.data if str(spec["res"]) == res)
            spec_data = np.dstack((wanted_res["flux_mean"], wanted_res["flux_var"]))
            ds = group.create_dataset(self.file_name, spec_data.shape, compression="gzip")
            ds.write_direct(spec_data)
            spec_datasets.append(ds)
        return spec_datasets

    def add_metadata(self, datasets):
        for ds in datasets:
            for key, value in dict(self.metadata).items():
                ds.attrs.create(key, value)


