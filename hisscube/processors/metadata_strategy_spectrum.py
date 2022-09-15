import os
from abc import ABC, abstractmethod

import h5py
import numpy as np

from hisscube.processors.metadata_strategy import MetadataStrategy
from hisscube.utils import astrometry
from hisscube.utils.astrometry import get_region_ref, NoCoverageFoundError, get_heal_path_from_coords
from hisscube.utils.config import Config
from hisscube.utils.io import H5Connector
from hisscube.utils.logging import HiSSCubeLogger
from hisscube.utils.photometry import Photometry


class SpectrumMetadataStrategy(ABC):
    def __init__(self, metadata_strategy: MetadataStrategy, config: Config, photometry: Photometry):
        self.metadata_strategy = metadata_strategy
        self.config = config
        self.photometry = photometry
        self.h5_connector: H5Connector = None
        self.logger = HiSSCubeLogger.logger

    @abstractmethod
    def write_parsed_spectrum_metadata(self, metadata, spectrum_length, fits_path, no_attrs, no_datasets):
        raise NotImplementedError

    @abstractmethod
    def get_resolution_groups(self, metadata, h5_connector):
        raise NotImplementedError

    @abstractmethod
    def link_spectra_to_images(self, h5_connector):
        raise NotImplementedError


class TreeSpectrumStrategy(SpectrumMetadataStrategy):

    def write_parsed_spectrum_metadata(self, metadata, spectrum_length, fits_path, no_attrs, no_datasets):
        file_name = os.path.basename(fits_path)
        res_grps = self._create_spectrum_index_tree(metadata, spectrum_length)
        spec_datasets = []
        if not no_datasets:
            spec_datasets = self._create_spec_datasets(file_name, res_grps)
        if not no_attrs:
            self.metadata_strategy.add_metadata(self.h5_connector, metadata, spec_datasets)

    def get_resolution_groups(self, metadata, h5_connector):
        self.h5_connector = h5_connector
        spatial_path = get_heal_path_from_coords(metadata, self.config, order=self.config.SPEC_SPAT_INDEX_ORDER)
        try:
            time = metadata["TAI"]
        except KeyError:
            time = metadata["MJD"]
        path = "/".join([spatial_path, str(time)])
        time_grp = self.h5_connector.file[path]
        for res_grp in time_grp:
            yield time_grp[res_grp]

    def link_spectra_to_images(self, h5_connector):
        self.h5_connector = h5_connector
        self._add_image_refs(h5_connector.file)

    def _create_spectrum_index_tree(self, metadata, spectrum_length):
        """
        Creates the index tree for a spectrum.
        Returns HDF5 group - the one where the spectrum dataset should be placed.
        -------

        """
        spec_grp = self.h5_connector.require_semi_sparse_cube_grp()
        spatial_grp = self._require_spectrum_spatial_grp_structure(metadata, spec_grp)
        time_grp = self._require_spectrum_time_grp(metadata, spatial_grp)
        res_grps = self._require_res_grps(time_grp, spectrum_length)
        return res_grps

    def _create_spec_datasets(self, file_name, parent_grp_list):
        spec_datasets = []
        for group in parent_grp_list:
            if self.config.C_BOOSTER:
                if "spectrum_dataset" in group:
                    raise RuntimeError(
                        "There is already an image dataset %s within this resolution group. Trying to insert image %s." % (
                            list(group["spectrum_dataset"]), file_name))
            elif len(group) > 0:
                raise RuntimeError(
                    "There is already a spectrum dataset %s within this resolution group. Trying to insert spectrum %s." % (
                        list(group), file_name))
            res = int(self.h5_connector.get_name(group).split('/')[-1])
            spec_data_shape = (res,) + (3,)
            ds = self.h5_connector.create_spectrum_h5_dataset(group, file_name, spec_data_shape)
            self.h5_connector.set_attr(ds, "mime-type", "spectrum")
            spec_datasets.append(ds)
        return spec_datasets

    def _require_spectrum_spatial_grp_structure(self, metadata, child_grp):
        """
        Creates the spatial index part for a spectrum. Takes the root group as parameter.
        Parameters
        ----------
        child_grp   HDF5 group

        Returns     HDF5 group
        -------

        """
        spectrum_coord = (metadata['PLUG_RA'], metadata['PLUG_DEC'])
        for order in range(self.config.SPEC_SPAT_INDEX_ORDER):
            child_grp = self.metadata_strategy.require_spatial_grp(self.h5_connector, order, child_grp, spectrum_coord)

        for img_zoom in range(self.config.IMG_ZOOM_CNT):
            self.h5_connector.require_dataset(child_grp, "image_cutouts_%d" % img_zoom,
                                              (self.config.MAX_CUTOUT_REFS,),
                                              dtype=h5py.regionref_dtype)

        return child_grp

    def _require_spectrum_time_grp(self, metadata, parent_grp):
        time = get_time_from_spectrum(metadata)
        grp = self.h5_connector.require_group(parent_grp, str(time), track_order=True)
        self.h5_connector.set_attr(grp, "type", "time")
        return grp

    def _require_res_grps(self, parent_grp, spectrum_length):
        res_grp_list = []
        x_lower_res = int(spectrum_length)
        for res_zoom in range(self.config.SPEC_ZOOM_CNT):
            res_grp_name = str(x_lower_res)
            grp = self.h5_connector.require_group(parent_grp, res_grp_name)
            self.h5_connector.set_attr(grp, "type", "resolution")
            self.h5_connector.set_attr(grp, "res_zoom", res_zoom)
            res_grp_list.append(grp)
            x_lower_res = int(x_lower_res / 2)
        return res_grp_list

    def _add_image_refs(self, h5_grp, depth=-1):
        if "type" in h5_grp.attrs and \
                h5_grp.attrs["type"] == "spatial" and \
                depth == self.config.SPEC_SPAT_INDEX_ORDER:
            spec_datasets = []
            time_grp_1st_spectrum = {}

            for child_grp in h5_grp.values():
                if isinstance(child_grp, h5py.Group) and child_grp.attrs["type"] == "time":
                    time_grp_1st_spectrum = child_grp  # we can take the first, all of the spectra have same coordinates here
                    break

            for res_grp in time_grp_1st_spectrum.values():
                for ds_name, ds in res_grp.items():
                    if ds_name.endswith("fits"):
                        spec_datasets.append(ds)
            self._add_image_refs_to_spectra(spec_datasets)
        else:
            if isinstance(h5_grp, h5py.Group):
                for child_grp in h5_grp.values():
                    if isinstance(child_grp, h5py.Group):
                        self._add_image_refs(child_grp, depth + 1)

    def _add_image_refs_to_spectra(self, spec_datasets):
        """
        Adds HDF5 Region references of image cut-outs to spectra attribute "image_cutouts". Throws NoCoverageFoundError
        if the cut-out does not span the whole cutout size for any reason.
        Parameters
        ----------
        spec_datasets   [HDF5 Datasets]

        Returns         [HDF5 Datasets]
        -------

        """

        image_refs = {}
        image_min_zoom_idx = 0
        metadata = self.h5_connector.read_serialized_fits_header(spec_datasets[0])
        for image_res_idx, image_ds in self._find_images_overlapping_spectrum(metadata):
            if not image_res_idx in image_refs:
                image_refs[image_res_idx] = []
            try:
                image_refs[image_res_idx].append(
                    get_region_ref(self.h5_connector, image_res_idx, image_ds, metadata,
                                   self.config.IMAGE_CUTOUT_SIZE))
                if image_res_idx > image_min_zoom_idx:
                    image_min_zoom_idx = image_res_idx
            except NoCoverageFoundError as e:
                self.logger.debug(
                    "No coverage found for spectrum %s and image %s, reason %s" % (spec_datasets[0], image_ds, str(e)))
                pass

        for res in image_refs:
            image_refs[res] = np.array(image_refs[res],
                                       dtype=h5py.regionref_dtype)
        for spec_zoom_idx, spec_ds in enumerate(spec_datasets):
            image_cutout_ds = spec_ds.parent.parent.parent[
                "image_cutouts_%d" % spec_zoom_idx]  # we write image cutout zoom equivalent to the spectral zoom
            if len(image_refs) > 0:
                if spec_zoom_idx > image_min_zoom_idx:
                    no_references = len(image_refs[image_min_zoom_idx])
                    image_cutout_ds[0:no_references] = image_refs[image_min_zoom_idx]
                else:
                    no_references = len(image_refs[spec_zoom_idx])
                    image_cutout_ds[0:no_references] = image_refs[spec_zoom_idx]
        return spec_datasets

    def _find_images_overlapping_spectrum(self, metadata):
        """Finds images in the HDF5 index structure that overlap the spectrum coordinate. Does so by constructing the
        whole heal_path string to the image and it to get the correct Group containing those images. Yields resolution
        index and the image dataset.

        Yields         (int, HDF5 dataset)
        -------
        """
        overlapping_pixel_paths = astrometry.get_potential_overlapping_image_spatial_paths(
            metadata,
            self.config.IMG_DIAMETER_ANG_MIN,
            self.config.IMG_SPAT_INDEX_ORDER)
        heal_paths = self._get_absolute_heal_paths(overlapping_pixel_paths)
        for heal_path in heal_paths:
            try:
                heal_path_group = self.h5_connector.file[heal_path]
                for time_grp in heal_path_group.values():
                    if isinstance(time_grp, h5py.Group):
                        for band_grp in time_grp.values():
                            if isinstance(band_grp, h5py.Group) and band_grp.attrs["type"] == "spectral":
                                for res_idx, res in enumerate(band_grp):
                                    res_grp = band_grp[res]
                                    for image_ds in res_grp.values():
                                        if image_ds.attrs["mime-type"] == "image":
                                            yield res_idx, image_ds
            except KeyError:
                pass

    def _get_absolute_heal_paths(self, overlapping_pixel_paths):
        for heal_path in overlapping_pixel_paths:
            absolute_path = "%s/%s" % (self.config.ORIG_CUBE_NAME, heal_path)
            yield absolute_path


class DatasetSpectrumStrategy(SpectrumMetadataStrategy):

    def link_spectra_to_images(self, h5_connector):
        pass

    def write_parsed_spectrum_metadata(self, metadata, spectrum_length, fits_path, no_attrs, no_datasets):
        pass

    def get_resolution_groups(self, metadata, h5_connector):
        pass


def get_time_from_spectrum(spec_header):
    try:
        time = spec_header["TAI"]
    except KeyError:
        time = spec_header["MJD"]
    return time
