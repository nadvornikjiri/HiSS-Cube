import os
from ast import literal_eval

import ujson

from hisscube.utils import astrometry
from hisscube.utils.astrometry import get_heal_path_from_coords
from hisscube.utils.io import get_path_patterns, get_image_header_dataset
from hisscube.utils.logging import HiSSCubeLogger, log_timing


class ImageMetadataProcessor:
    def __init__(self, config, metadata_handler):
        self.h5_connector = None
        self.metadata_processor = metadata_handler
        self.config = config
        self.logger = HiSSCubeLogger.logger
        self.img_cnt = 0

    def set_connector(self, h5_connector):
        self.h5_connector = h5_connector
        self.metadata_processor.h5_connector = h5_connector

    def require_res_grps(self, parent_grp):
        res_grp_list = []
        x_lower_res = int(self.metadata_processor.metadata["NAXIS1"])
        y_lower_res = int(self.metadata_processor.metadata["NAXIS2"])
        for res_zoom in range(self.config.IMG_ZOOM_CNT):
            res_grp_name = str((x_lower_res, y_lower_res))
            grp = self.h5_connector.require_group(parent_grp, res_grp_name)
            self.h5_connector.set_attr(grp, "type", "resolution")
            self.h5_connector.set_attr(grp, "res_zoom", res_zoom)
            res_grp_list.append(grp)
            x_lower_res = int(x_lower_res / 2)
            y_lower_res = int(y_lower_res / 2)
        return res_grp_list

    def update_image_headers(self, h5_connector, image_path, image_pattern=None):
        self.set_connector(h5_connector)
        image_pattern, spectra_pattern = get_path_patterns(self.config, image_pattern, None)
        try:
            self.img_cnt = self.h5_connector.file.attrs["image_count"]  # header datasets not created yet
        except KeyError:
            self.img_cnt = 0
            self.metadata_processor.create_fits_header_datasets()
        image_header_ds = get_image_header_dataset(h5_connector)
        self.img_cnt += self.metadata_processor.write_fits_headers(image_header_ds, image_header_ds.dtype, image_path,
                                                                   image_pattern,
                                                                   self.config.LIMIT_IMAGE_COUNT, offset=self.img_cnt)
        self.h5_connector.file.attrs["image_count"] = self.img_cnt

    def create_image_index_tree(self):
        """
        Creates the index tree for an image.
        Returns HDF5 group - the one where the image dataset should be placed.
        -------

        """
        cube_grp = self.h5_connector.require_raw_cube_grp()
        spatial_grp = self.require_image_spatial_grp_structure(cube_grp)
        time_grp = self.require_image_time_grp(spatial_grp)
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
        image_coords = astrometry.get_image_center_coords(self.metadata_processor.metadata)

        parent_grp = orig_parent
        for order in range(self.config.IMG_SPAT_INDEX_ORDER):
            parent_grp = self.metadata_processor.require_spatial_grp(order, parent_grp, image_coords)
            if order == self.config.IMG_SPAT_INDEX_ORDER - 1:
                return parent_grp

    def require_image_time_grp(self, parent_grp):
        tai_time = self.metadata_processor.metadata["TAI"]
        grp = self.h5_connector.require_group(parent_grp, str(tai_time))
        self.h5_connector.set_attr(grp, "type", "time")
        return grp

    def require_image_spectral_grp(self, parent_grp):
        grp = self.h5_connector.require_group(parent_grp, str(
            self.metadata_processor.photometry.filter_midpoints[self.metadata_processor.metadata["FILTER"]]),
                                              track_order=True)
        self.h5_connector.set_attr(grp, "type", "spectral")
        return grp

    def write_images_metadata(self, h5_connector, no_attrs=False, no_datasets=False):
        self.set_connector(h5_connector)
        fits_headers = self.h5_connector.file["/fits_images_metadata"]
        self.write_image_metadata_from_cache(h5_connector, fits_headers, no_attrs, no_datasets)

    def write_image_metadata_from_cache(self, h5_connector, fits_headers, no_attrs, no_datasets):
        self.img_cnt = 0
        self.h5_connector.fits_total_cnt = 0
        for fits_path, header in fits_headers:
            if not fits_path:  # end of data
                break
            self.write_metadata_from_header(h5_connector, fits_path, header, no_attrs, no_datasets)
            if self.img_cnt >= self.config.LIMIT_IMAGE_COUNT:
                break
        self.h5_connector.set_attr(self.h5_connector.file, "image_count", self.img_cnt)

    @log_timing("process_image_metadata")
    def write_metadata_from_header(self, h5_connector, fits_path, header, no_attrs, no_datasets):
        self.set_connector(h5_connector)
        fits_path = fits_path.decode('utf-8')
        if self.img_cnt % 100 == 0 and self.img_cnt / 100 > 0:
            self.logger.info("Image cnt: %05d" % self.img_cnt)
        try:
            self.write_image_metadata(h5_connector, fits_path, header, no_attrs=no_attrs, no_datasets=no_datasets)
            self.img_cnt += 1
            self.h5_connector.fits_total_cnt += 1
        except RuntimeError as e:
            self.logger.warning(
                "Unable to ingest image %s, message: %s" % (fits_path, str(e)))
            raise e

    def write_image_metadata(self, h5_connector, fits_path, fits_header, no_attrs=False, no_datasets=False):
        self.set_connector(h5_connector)
        self.metadata_processor.metadata = ujson.loads(fits_header)
        self.write_parsed_image_metadata(fits_path, no_attrs, no_datasets)



    def write_parsed_image_metadata(self, fits_path, no_attrs, no_datasets):
        img_datasets = []
        self.metadata_processor.file_name = os.path.basename(fits_path)
        res_grps = self.create_image_index_tree()
        if not no_datasets:
            img_datasets = self.create_img_datasets(res_grps)
        if not no_attrs:
            self.metadata_processor.add_metadata(img_datasets)

    def get_resolution_groups(self, h5_connector):
        self.set_connector(h5_connector)
        reference_coord = astrometry.get_image_center_coords(self.metadata_processor.metadata)
        spatial_path = get_heal_path_from_coords(self.metadata_processor.metadata, self.config, ra=reference_coord[0],
                                                 dec=reference_coord[1])
        tai_time = self.metadata_processor.metadata["TAI"]
        spectral_midpoint = self.metadata_processor.photometry.filter_midpoints[self.metadata_processor.metadata["FILTER"]]
        path = "/".join([spatial_path, str(tai_time), str(spectral_midpoint)])
        spectral_grp = self.h5_connector.file[path]
        for res_grp in spectral_grp:
            yield spectral_grp[res_grp]

    def create_img_datasets(self, parent_grp_list):
        img_datasets = []
        for group in parent_grp_list:
            if self.config.C_BOOSTER:
                if "image_dataset" in group:
                    raise RuntimeError(
                        "There is already an image dataset %s within this resolution group. Trying to insert image %s." % (
                            list(group["image_dataset"]), self.metadata_processor.file_name))
            elif len(group) > 0:
                raise RuntimeError(
                    "There is already an image dataset %s within this resolution group. Trying to insert image %s." % (
                        list(group), self.metadata_processor.file_name))
            res_tuple = self.h5_connector.get_name(group).split('/')[-1]
            img_data_shape = tuple(reversed(literal_eval(res_tuple))) + (2,)
            ds = self.h5_connector.create_image_h5_dataset(group, self.metadata_processor.file_name, img_data_shape)
            self.h5_connector.set_attr(ds, "mime-type", "image")
            img_datasets.append(ds)
        return img_datasets


