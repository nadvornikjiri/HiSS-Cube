import os
import pathlib

import fitsio
import h5py.h5p

from hisscube import astrometry
from ast import literal_eval as make_tuple
import numpy as np

from hisscube.H5Handler import H5Handler
from timeit import default_timer as timer

from hisscube.fitstools import read_primary_header_quick, read_header_from_path


class ImageWriter(H5Handler):

    def __init__(self, h5_file=None, h5_path=None, timings_csv="timings.csv"):
        super().__init__(h5_file, h5_path, timings_csv)

    def ingest_image(self, image_path):
        """
        Method that writes an image to the opened HDF5 file (self.f).
        Parameters
        ----------
        image_path  String

        Returns     HDF5 Dataset (already written to the file)
        -------

        """
        self.write_image_metadata(image_path)
        self.metadata, self.data = self.cube_utils.get_multiple_resolution_image(image_path,
                                                                                 self.config.getint("Handler",
                                                                                                    "IMG_ZOOM_CNT"))
        img_datasets = self.write_img_datasets()
        return img_datasets

    def create_image_index_tree(self):
        """
        Creates the index tree for an image.
        Returns HDF5 group - the one where the image dataset should be placed.
        -------

        """
        cube_grp = self.require_raw_cube_grp()
        spatial_grps = self.require_image_spatial_grp_structure(cube_grp)
        time_grp = self.require_image_time_grp(spatial_grps[0])
        self.add_hard_links(spatial_grps[1:], time_grp)
        img_spectral_grp = self.require_image_spectral_grp(time_grp)
        res_grps = self.require_res_grps(img_spectral_grp)
        return res_grps

    def require_image_spatial_grp_structure(self, parent_grp):
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
        boundaries = astrometry.get_boundary_coords(self.metadata)
        leaf_grp_set = []
        for coord in boundaries:
            parent_grp = orig_parent
            for order in range(self.IMG_SPAT_INDEX_ORDER):
                parent_grp = self.require_spatial_grp(order, parent_grp, coord)
                if order == self.IMG_SPAT_INDEX_ORDER - 1:
                    # only return each leaf group once.
                    if len(leaf_grp_set) == 0 or \
                            not (any(self.get_name(grp) == self.get_name(parent_grp) for grp in leaf_grp_set)):
                        leaf_grp_set.append(parent_grp)
        return leaf_grp_set

    def require_image_time_grp(self, parent_grp):
        tai_time = self.metadata["TAI"]
        grp = self.require_group(parent_grp, str(tai_time))
        self.set_attr(grp, "type", "time")
        return grp

    def require_image_spectral_grp(self, parent_grp):
        grp = self.require_group(parent_grp, str(self.cube_utils.filter_midpoints[self.metadata["FILTER"]]),
                                 track_order=True)
        self.set_attr(grp, "type", "spectral")
        return grp

    def create_img_datasets(self, parent_grp_list):
        img_datasets = []
        for group in parent_grp_list:
            if self.C_BOOSTER:
                if "image_dataset" in group:
                    raise ValueError(
                        "There is already an image dataset %s within this resolution group. Trying to insert image %s." % (
                            list(group), self.file_name))
            elif len(group) > 0:
                raise ValueError(
                    "There is already an image dataset %s within this resolution group. Trying to insert image %s." % (
                        list(group), self.file_name))
            res_tuple = self.get_name(group).split('/')[-1]
            img_data_shape = tuple(reversed(make_tuple(res_tuple))) + (2,)
            ds = self.create_image_h5_dataset(group, img_data_shape)
            self.set_attr(ds, "mime-type", "image")
            img_datasets.append(ds)
        return img_datasets

    def create_image_h5_dataset(self, group, img_data_shape):
        dcpl, space, img_data_dtype = self.get_property_list(img_data_shape)
        if self.CHUNK_SIZE:
            dcpl.set_chunk(make_tuple(self.CHUNK_SIZE))
        dsid = h5py.h5d.create(group.id, self.file_name.encode(), img_data_dtype, space, dcpl=dcpl)
        ds = h5py.Dataset(dsid)
        return ds

    def write_images_metadata(self, image_folder, image_pattern, no_attrs=False, no_datasets=False):
        start = timer()
        check = 100
        for fits_path in pathlib.Path(image_folder).rglob(
                image_pattern):
            if self.img_cnt % check == 0 and self.img_cnt / check > 0:
                end = timer()
                self.logger.info("100 images done in %.4fs" % (end - start))
                self.log_metadata_csv_timing(end - start)
                start = end
                self.logger.info("Image cnt: %05d" % self.img_cnt)
            self.write_image_metadata(fits_path, no_attrs, no_datasets)
            self.img_cnt += 1
            if self.img_cnt >= self.LIMIT_IMAGE_COUNT:
                break
        self.set_attr(self.f, "image_count", self.img_cnt)

    def write_image_metadata(self, fits_path, no_attrs=False, no_datasets=False):
        self.ingest_type = "image"
        self.image_path_list.append(str(fits_path))
        self.metadata = read_header_from_path(fits_path)
        self.file_name = os.path.basename(fits_path)
        res_grps = self.create_image_index_tree()
        if not no_datasets:
            img_datasets = self.create_img_datasets(res_grps)
        if not no_attrs:
            self.add_metadata(img_datasets)

    def write_img_datasets(self, no_attrs=False, no_datasets=False):
        res_grp_list = self.get_image_resolution_groups()
        img_datasets = []
        for group in res_grp_list:
            res_tuple = group.name.split('/')[-1]
            wanted_res = next(img for img in self.data if str(tuple(img["res"])) == res_tuple)  # parsing 2D resolution
            img_data = np.dstack((wanted_res["flux_mean"], wanted_res["flux_sigma"]))
            img_data[img_data == np.inf] = np.nan
            if self.FLOAT_COMPRESS:
                img_data = self.float_compress(img_data)
            ds = group[self.file_name]
            ds.write_direct(img_data)
            img_datasets.append(ds)
        return img_datasets

    def get_image_resolution_groups(self):
        reference_coord = astrometry.get_boundary_coords(self.metadata)[0]
        spatial_path = self.get_heal_path_from_coords(ra=reference_coord[0], dec=reference_coord[1])
        tai_time = self.metadata["TAI"]
        spectral_midpoint = self.cube_utils.filter_midpoints[self.metadata["FILTER"]]
        path = "/".join([spatial_path, str(tai_time), str(spectral_midpoint)])
        spectral_grp = self.f[path]
        for res_grp in spectral_grp:
            yield spectral_grp[res_grp]

    def get_name(self, grp):
        return grp.name
