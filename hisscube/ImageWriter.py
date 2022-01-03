import os
import pathlib

import fitsio
import h5py.h5p

from hisscube import astrometry
from ast import literal_eval as make_tuple
import numpy as np

from hisscube.H5Handler import H5Handler
from timeit import default_timer as timer
import csv


class ImageWriter(H5Handler):

    def __init__(self, h5_file=None, h5_path=None):
        super().__init__(h5_file, h5_path)
        self.img_cnt = 0

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
            for order in range(self.config.getint("Handler", "IMG_SPAT_INDEX_ORDER")):
                parent_grp = self.require_spatial_grp(order, parent_grp, coord)
                if order == self.config.getint("Handler", "IMG_SPAT_INDEX_ORDER") - 1:
                    # only return each leaf group once.
                    if len(leaf_grp_set) == 0 or \
                            not (any(grp.name == parent_grp.name for grp in leaf_grp_set)):
                        leaf_grp_set.append(parent_grp)
        return leaf_grp_set

    def require_image_time_grp(self, parent_grp):
        tai_time = self.metadata["TAI"]
        grp = self.require_group(parent_grp, str(tai_time))
        grp.attrs["type"] = "time"
        return grp

    def require_image_spectral_grp(self, parent_grp):
        grp = self.require_group(parent_grp, str(self.cube_utils.filter_midpoints[self.metadata["FILTER"]]),
                                 track_order=True)
        grp.attrs["type"] = "spectral"
        return grp

    def create_img_datasets(self, parent_grp_list):
        img_datasets = []
        for group in parent_grp_list:
            res_tuple = group.name.split('/')[-1]
            img_data_shape = tuple(reversed(make_tuple(res_tuple))) + (2,)
            dcpl, space, img_data_dtype = self.get_property_list(img_data_shape)
            if self.config.get("Handler", "CHUNK_SIZE"):
                dcpl.set_chunk(make_tuple(self.config.get("Handler", "CHUNK_SIZE")))
            dsid = h5py.h5d.create(group.id, self.file_name.encode(), img_data_dtype, space, dcpl=dcpl)
            ds = h5py.Dataset(dsid)
            ds.attrs["mime-type"] = "image"
            img_datasets.append(ds)
        return img_datasets

    def write_lower_res_wcs(self, ds, res_idx=0):
        """
        Modifies the FITS WCS parameters for lower resolutions of the image so it is still correct.
        Parameters
        ----------
        ds      HDF5 dataset
        res_idx int

        Returns
        -------

        """
        w = astrometry.get_optimized_wcs(self.metadata)
        w.wcs.crpix /= 2 ** res_idx  # shift center of the image
        w.wcs.cd *= 2 ** res_idx  # change the pixel scale
        image_fits_header = ds.attrs
        image_fits_header["CRPIX1"], image_fits_header["CRPIX2"] = w.wcs.crpix
        [[image_fits_header["CD1_1"], image_fits_header["CD1_2"]],
         [image_fits_header["CD2_1"], image_fits_header["CD2_2"]]] = w.wcs.cd
        image_fits_header["CRVAL1"], image_fits_header["CRVAL2"] = w.wcs.crval
        image_fits_header["CTYPE1"], image_fits_header["CTYPE2"] = w.wcs.ctype

    def write_images_metadata(self, image_folder, image_pattern):
        start = timer()
        check = 100
        for fits_path in pathlib.Path(image_folder).rglob(
                image_pattern) and self.img_cnt < 50000:  # TODO remove test number
            if self.img_cnt % check == 0 and self.img_cnt / check > 0:
                end = timer()
                self.logger.info("100 images done in %.4fs" % (end - start))
                self.log_csv_timing(end - start)
                start = end
                self.logger.info("Image cnt: %05d" % self.img_cnt)
            self.write_image_metadata(fits_path)
            self.img_cnt += 1
        self.csv_file.close()
        self.f.attrs["image_count"] = self.img_cnt

    def write_image_metadata(self, fits_path):
        self.ingest_type = "image"
        self.image_path_list.append(str(fits_path))
        self.metadata = fitsio.read_header(fits_path)
        self.file_name = os.path.basename(fits_path)
        res_grps = self.create_image_index_tree()
        img_datasets = self.create_img_datasets(res_grps)
        self.add_metadata(img_datasets)

    def write_img_datasets(self):
        res_grp_list = self.get_image_resolution_groups()
        img_datasets = []
        for group in res_grp_list:
            res_tuple = group.name.split('/')[-1]
            wanted_res = next(img for img in self.data if str(tuple(img["res"])) == res_tuple)  # parsing 2D resolution
            img_data = np.dstack((wanted_res["flux_mean"], wanted_res["flux_sigma"]))
            img_data[img_data == np.inf] = np.nan
            if self.config.getboolean("Writer", "FLOAT_COMPRESS"):
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

    def log_csv_timing(self, time):
        self.timings_logger.writerow([self.img_cnt, self.grp_cnt, time])
