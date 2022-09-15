import os
from abc import ABC, abstractmethod
from ast import literal_eval

from hisscube.processors.metadata_strategy import MetadataStrategy
from hisscube.utils import astrometry
from hisscube.utils.astrometry import get_heal_path_from_coords
from hisscube.utils.config import Config
from hisscube.utils.io import H5Connector
from hisscube.utils.nexus import set_nx_data
from hisscube.utils.photometry import Photometry


class ImageMetadataStrategy(ABC):
    def __init__(self, metadata_strategy: MetadataStrategy, config: Config, photometry: Photometry):
        self.metadata_strategy = metadata_strategy
        self.config = config
        self.photometry = photometry
        self.h5_connector: H5Connector = None

    @abstractmethod
    def get_resolution_groups(self, metadata, h5_connector):
        raise NotImplementedError

    @abstractmethod
    def write_parsed_image_metadata(self, metadata, fits_path, no_attrs, no_datasets):
        raise NotImplementedError


class TreeImageStrategy(ImageMetadataStrategy):

    def write_parsed_image_metadata(self, metadata, fits_path, no_attrs, no_datasets):
        img_datasets = []
        file_name = os.path.basename(fits_path)
        res_grps = self._create_image_index_tree(metadata)
        if not no_datasets:
            img_datasets = self._create_img_datasets(file_name, res_grps)
        if not no_attrs:
            self.metadata_strategy.add_metadata(self.h5_connector, metadata, img_datasets)

    def get_resolution_groups(self, metadata, h5_connector):
        h5_connector = h5_connector
        reference_coord = astrometry.get_image_center_coords(metadata)
        spatial_path = get_heal_path_from_coords(metadata, self.config, ra=reference_coord[0],
                                                 dec=reference_coord[1])
        tai_time = metadata["TAI"]
        spectral_midpoint = self.photometry.filter_midpoints[metadata["FILTER"]]
        path = "/".join([spatial_path, str(tai_time), str(spectral_midpoint)])
        spectral_grp = h5_connector.file[path]
        for res_grp in spectral_grp:
            yield spectral_grp[res_grp]

    def _create_image_index_tree(self, metadata):
        """
        Creates the index tree for an image.
        Returns HDF5 group - the one where the image dataset should be placed.
        -------

        """
        cube_grp = self.h5_connector.require_semi_sparse_cube_grp()
        spatial_grp = self._require_image_spatial_grp_structure(metadata, cube_grp)
        time_grp = self._require_image_time_grp(metadata, spatial_grp)
        img_spectral_grp = self.require_image_spectral_grp(metadata, time_grp)
        res_grps = self._require_res_grps(metadata, img_spectral_grp)
        return res_grps

    def _create_img_datasets(self, file_name, parent_grp_list):
        img_datasets = []
        for group in parent_grp_list:
            if self.config.C_BOOSTER:
                if "image_dataset" in group:
                    raise RuntimeError(
                        "There is already an image dataset %s within this resolution group. Trying to insert image %s." % (
                            list(group["image_dataset"]), file_name))
            elif len(group) > 0:
                raise RuntimeError(
                    "There is already an image dataset %s within this resolution group. Trying to insert image %s." % (
                        list(group), file_name))
            res_tuple = self.h5_connector.get_name(group).split('/')[-1]
            img_data_shape = tuple(reversed(literal_eval(res_tuple))) + (2,)
            ds = self.h5_connector.create_image_h5_dataset(group, file_name, img_data_shape)
            self.h5_connector.set_attr(ds, "mime-type", "image")
            self.h5_connector.set_attr(ds, "interpretation", "image")
            img_datasets.append(ds)
        return img_datasets

    def _require_image_spatial_grp_structure(self, metadata, parent_grp):
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
        image_coords = astrometry.get_image_center_coords(metadata)

        parent_grp = orig_parent
        for order in range(self.config.IMG_SPAT_INDEX_ORDER):
            parent_grp = self.metadata_strategy.require_spatial_grp(self.h5_connector, order, parent_grp, image_coords)
            if order == self.config.IMG_SPAT_INDEX_ORDER - 1:
                return parent_grp

    def _require_image_time_grp(self, metadata, parent_grp):
        tai_time = metadata["TAI"]
        grp = self.h5_connector.require_group(parent_grp, str(tai_time))
        self.h5_connector.set_attr(grp, "type", "time")
        return grp

    def require_image_spectral_grp(self, metadata, parent_grp):
        grp = self.h5_connector.require_group(parent_grp, str(
            self.photometry.filter_midpoints[metadata["FILTER"]]),
                                              track_order=True)
        self.h5_connector.set_attr(grp, "type", "spectral")
        return grp

    def _require_res_grps(self, metadata, parent_grp):
        res_grp_list = []
        x_lower_res = int(metadata["NAXIS1"])
        y_lower_res = int(metadata["NAXIS2"])
        for res_zoom in range(self.config.IMG_ZOOM_CNT):
            res_grp_name = str((x_lower_res, y_lower_res))
            grp = self.h5_connector.require_group(parent_grp, res_grp_name)
            self.h5_connector.set_attr(grp, "type", "resolution")
            self.h5_connector.set_attr(grp, "res_zoom", res_zoom)
            set_nx_data(grp, self.h5_connector)
            res_grp_list.append(grp)
            x_lower_res = int(x_lower_res / 2)
            y_lower_res = int(y_lower_res / 2)
        return res_grp_list


class DatasetImageStrategy(ImageMetadataStrategy):

    def write_parsed_image_metadata(self, metadata, fits_path, no_attrs, no_datasets):
        pass

    def get_resolution_groups(self, metadata, h5_connector):
        pass
