import os
import traceback
from abc import ABC, abstractmethod, ABCMeta
from json import JSONDecodeError
from pathlib import Path

import fitsio
import h5py
import numpy as np
import ujson
from tqdm.auto import tqdm

from hisscube.processors.data import float_compress
from hisscube.processors.metadata_strategy import MetadataStrategy, require_zoom_grps
from hisscube.processors.metadata_strategy_dataset import get_cutout_data_datasets, \
    get_cutout_error_datasets, get_cutout_metadata_datasets, get_data_datasets, get_error_datasets, get_index_datasets, \
    get_metadata_datasets, get_healpix_id, get_dataset_resolution_groups, write_dataset, recreate_additional_datasets, \
    get_wcs_datasets
from hisscube.processors.metadata_strategy_tree import require_spatial_grp, TreeStrategy
from hisscube.utils import astrometry
from hisscube.utils.astrometry import NoCoverageFoundError, get_heal_path_from_coords, \
    get_spectrum_center_coords, get_overlapping_healpix_pixel_ids, get_cutout_bounds, is_cutout_whole
from hisscube.utils.config import Config
from hisscube.utils.io import H5Connector, get_spectrum_header_dataset
from hisscube.utils.logging import HiSSCubeLogger, log_timing, wrap_tqdm
from hisscube.utils.nexus import set_nx_interpretation, set_nx_axes
from hisscube.utils.photometry import Photometry


class SpectrumMetadataStrategy(ABC, metaclass=ABCMeta):
    def __init__(self, metadata_strategy: MetadataStrategy, config: Config, photometry: Photometry, logger: HiSSCubeLogger):
        self.metadata_strategy = metadata_strategy
        self.config = config
        self.photometry = photometry
        self.h5_connector: H5Connector = None
        self.logger = logger
        self.spec_cnt = 0

    def write_metadata_multiple(self, h5_connector, no_attrs=False, no_datasets=False, range_min=None, range_max=None,
                                batch_size=None):
        self._set_connector(h5_connector)
        fits_headers = get_spectrum_header_dataset(h5_connector)
        self._write_metadata_from_cache(h5_connector, fits_headers, no_attrs, no_datasets, range_min, range_max,
                                        batch_size)

    def write_metadata(self, h5_connector, fits_path, fits_header, no_attrs=False, no_datasets=False, offset=0,
                       batch_size=None):
        self._set_connector(h5_connector)
        metadata = ujson.loads(fits_header)
        self._write_parsed_metadata(metadata, fits_path, no_attrs, no_datasets, offset, batch_size)

    def _write_metadata_from_cache(self, h5_connector, fits_headers, no_attrs, no_datasets, range_min=None,
                                   range_max=None, batch_size=None):
        self.spec_cnt = 0
        self.h5_connector.fits_total_cnt = 0
        if not range_min:
            range_min = 0
        if not range_max:
            range_max = len(fits_headers)
        if not batch_size:
            batch_size = len(fits_headers)
        headers_batch = fits_headers[range_min:range_max]
        iterator = wrap_tqdm(headers_batch, self.config.MPIO, self.__class__.__name__, self.config)
        for fits_path, header in iterator:
            self._write_metadata_from_header(h5_connector, fits_path, header, no_attrs, no_datasets, range_min,
                                             batch_size)

    # @log_timing("process_spectrum_metadata")
    def _write_metadata_from_header(self, h5_connector, fits_path, header, no_attrs, no_datasets, offset=0,
                                    batch_size=None):
        fits_path = fits_path.decode('utf-8')
        self._set_connector(h5_connector)
        try:
            self.write_metadata(h5_connector, fits_path, header, no_attrs, no_datasets, offset, batch_size)
            self.spec_cnt += 1
            self.h5_connector.fits_total_cnt += 1
        except ValueError as e:
            self.logger.warning(
                "Unable to ingest spectrum %s, message: %s" % (fits_path, str(e)))

    def _write_image_cutouts(self, spec_zoom, image_cutout_ds, image_max_zoom_idx, image_refs, range_min=None,
                             batch_i=None, batch_size=None, buffer_idx=None):
        no_references = 0
        if spec_zoom > image_max_zoom_idx:
            if len(image_refs) > 0:
                no_references = len(image_refs[image_max_zoom_idx])
            self._write_image_cutout(image_cutout_ds, image_max_zoom_idx, image_refs, no_references, range_min, batch_i,
                                     batch_size, buffer_idx)
        else:
            if len(image_refs) > 0:
                no_references = len(image_refs[spec_zoom])
            self._write_image_cutout(image_cutout_ds, spec_zoom, image_refs, no_references, range_min, batch_i,
                                     batch_size, buffer_idx)

    def _set_connector(self, h5_connector):
        self.h5_connector = h5_connector

    @staticmethod
    def convert_refs_to_array(image_refs, dtype=None):
        for res in image_refs:
            image_refs[res] = np.array(image_refs[res], dtype=dtype)

    @abstractmethod
    def get_resolution_groups(self, metadata, h5_connector):
        raise NotImplementedError

    @abstractmethod
    def link_spectra_to_images(self, h5_connector, min_range=None, max_range=None, batch_size=None,
                               image_db_index=None, image_wcs_data=None):
        raise NotImplementedError

    @abstractmethod
    def write_datasets(self, res_grp_list, data, file_name, offset, batch_i, batch_size, coordinates=None):
        raise NotImplementedError

    @abstractmethod
    def _write_parsed_metadata(self, metadata, fits_path, no_attrs, no_datasets, offset, batch_size):
        raise NotImplementedError

    @abstractmethod
    def _write_image_cutout(self, image_cutout_ds, zoom, image_refs, no_references, range_min=None, batch_i=None,
                            batch_size=None, buffer_idx=None):
        pass

    def clear_buffers(self):
        pass

    def recreate_link_datasets(self, h5_connector, spec_count, spec_zoom_groups):
        pass


class TreeSpectrumStrategy(SpectrumMetadataStrategy):
    def __init__(self, metadata_strategy: TreeStrategy, config: Config, photometry: Photometry):
        super().__init__(metadata_strategy, config, photometry)

    def get_resolution_groups(self, metadata, h5_connector: H5Connector):
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

    def write_datasets(self, res_grp_list, data, file_name, offset, batch_i, batch_size, coordinates=None):
        spec_datasets = []
        for group in res_grp_list:
            res = group.name.split('/')[-1]
            wanted_res = next(spec for spec in data if str(spec["zoom_idx"]) == res)
            spec_data = np.column_stack((wanted_res["wl"], wanted_res["flux_mean"], wanted_res["flux_sigma"]))
            spec_data[spec_data == np.inf] = np.nan
            if self.config.FLOAT_COMPRESS:
                spec_data = float_compress(spec_data)
            ds = group[file_name]
            ds.write_direct(spec_data)
            spec_datasets.append(ds)
        return spec_datasets

    def link_spectra_to_images(self, h5_connector, min_range=None, max_range=None, batch_size=None,
                               image_db_index=None, image_wcs_data=None):
        self.h5_connector = h5_connector
        self._add_image_refs(h5_connector.file)

    def _write_parsed_metadata(self, metadata, fits_path, no_attrs, no_datasets, offset, batch_size):
        if self.config.APPLY_REBIN is False:
            spectrum_length = fitsio.read_header(fits_path, 1)["NAXIS2"]
        else:
            spectrum_length = self.config.REBIN_SAMPLES
        file_name = os.path.basename(fits_path)
        res_grps = self._create_index_tree(metadata, spectrum_length)
        spec_datasets = []
        if not no_datasets:
            spec_datasets = self._create_spec_datasets(file_name, res_grps)
        if not no_attrs:
            self.metadata_strategy.add_metadata(self.h5_connector, metadata, spec_datasets)

    def _create_index_tree(self, metadata, spectrum_length):
        """
        Creates the index tree for a spectrum.
        Returns HDF5 group - the one where the spectrum dataset should be placed.
        -------

        """
        spec_grp = self.h5_connector.require_semi_sparse_cube_grp()
        spatial_grp = self._require_spatial_grp_structure(metadata, spec_grp)
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
            if file_name not in group:
                ds = self.h5_connector.create_spectrum_h5_dataset(group, file_name, spec_data_shape)
            else:
                ds = group[file_name]
            self.h5_connector.set_attr(ds, "mime-type", "spectrum")
            spec_datasets.append(ds)
        return spec_datasets

    def _require_spatial_grp_structure(self, metadata, child_grp):
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
            child_grp = require_spatial_grp(self.h5_connector, order, child_grp, spectrum_coord)

        for img_zoom in range(self.config.IMG_ZOOM_CNT):
            self.h5_connector.require_dataset(child_grp, "image_cutouts_%d" % img_zoom,
                                              (self.config.MAX_CUTOUT_REFS,),
                                              dtype=h5py.regionref_dtype)

        return child_grp

    def _require_spectrum_time_grp(self, metadata, parent_grp):
        time = get_spectrum_time(metadata)
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
        if self.is_lowest_spatial_depth(depth, h5_grp):
            time_grp_1st_spectrum = self.get_spectrum_group(h5_grp)
            spec_datasets = self.get_spec_datasets(time_grp_1st_spectrum)
            self._add_image_refs_to_spectra(spec_datasets)
        else:
            self.walk_groups(depth, h5_grp)

    def get_region_ref_from_image(self, image_ds, image_max_zoom_idx, image_refs, image_res_idx, metadata,
                                  spec_datasets):
        if image_res_idx not in image_refs:
            image_refs[image_res_idx] = []
        try:
            image_refs[image_res_idx].append(
                self._get_region_ref(image_res_idx, image_ds, metadata,
                                     self.config.IMAGE_CUTOUT_SIZE))
            if image_res_idx > image_max_zoom_idx:
                image_max_zoom_idx = image_res_idx
        except NoCoverageFoundError as e:
            self.logger.debug(
                "No coverage found for spectrum %s and image %s, reason %s" % (spec_datasets[0], image_ds, str(e)))
            pass
        return image_max_zoom_idx, image_refs

    def _add_image_refs_to_spectra(self, spec_datasets_multiple_zoom):
        """
        Adds HDF5 Region references of image cut-outs to spectra attribute "sparse_cube". Throws NoCoverageFoundError
        if the cut-out does not span the whole cutout path_size for any reason.
        Parameters
        ----------
        spec_metadata_datasets_multiple_zoom   [HDF5 Datasets]

        Returns         [HDF5 Datasets]
        -------

        """

        image_refs = {}
        image_max_zoom_idx = 0
        original_resolution_dataset = spec_datasets_multiple_zoom[0]
        metadata = self.h5_connector.read_serialized_fits_header(original_resolution_dataset, idx=self.spec_cnt)
        for image_res_idx, image_ds in self._find_images_overlapping_spectrum(metadata):
            image_max_zoom_idx, image_refs = self.get_region_ref_from_image(image_ds, image_max_zoom_idx, image_refs,
                                                                            image_res_idx,
                                                                            metadata, spec_datasets_multiple_zoom)
        self.convert_refs_to_array(image_refs)
        for spec_zoom_idx, spec_ds in enumerate(spec_datasets_multiple_zoom):
            image_cutout_ds = self.get_image_cutout_ds(spec_ds, spec_zoom_idx)
            self._write_image_cutouts(spec_zoom_idx, image_cutout_ds, image_max_zoom_idx, image_refs)
        self.spec_cnt += 1
        return spec_datasets_multiple_zoom

    def walk_groups(self, depth, h5_grp):
        if isinstance(h5_grp, h5py.Group):
            for child_grp in h5_grp.values():
                if isinstance(child_grp, h5py.Group):
                    self._add_image_refs(child_grp, depth + 1)

    def is_lowest_spatial_depth(self, depth, h5_grp):
        return "type" in h5_grp.attrs and h5_grp.attrs[
            "type"] == "spatial" and depth == self.config.SPEC_SPAT_INDEX_ORDER

    @staticmethod
    def get_spec_datasets(time_grp_1st_spectrum):
        spec_datasets = []
        for res_grp in time_grp_1st_spectrum.values():
            for ds_name, ds in res_grp.items():
                if ds_name.endswith("fits"):
                    spec_datasets.append(ds)
        return spec_datasets

    @staticmethod
    def get_spectrum_group(h5_grp):
        time_grp_1st_spectrum = {}
        for child_grp in h5_grp.values():
            if isinstance(child_grp, h5py.Group) and child_grp.attrs["type"] == "time":
                time_grp_1st_spectrum = child_grp  # we can take the first, all of the spectra have same coordinates here
                break
        return time_grp_1st_spectrum

    def _write_image_cutout(self, image_cutout_ds, zoom, image_refs, no_references, range_min=None, batch_i=None,
                            batch_size=None, buffer_idx=None):
        image_cutout_ds[0:no_references] = image_refs[zoom]

    @staticmethod
    def get_image_cutout_ds(spec_ds, spec_zoom_idx):
        image_cutout_ds = spec_ds.parent.parent.parent[
            "image_cutouts_%d" % spec_zoom_idx]  # we write image cutout zoom_idx equivalent to the spectral zoom_idx
        return image_cutout_ds

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
            self.config.IMG_SPAT_INDEX_ORDER,
            self.config.IMG_X_SIZE_ANG_MIN,
            self.config.IMG_Y_SIZE_ANG_MIN)
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

    def _get_region_ref(self, image_zoom, image_ds, spec_fits_header, image_cutout_size):
        """
        Gets the region reference for a given resolution from an ds.

        Parameters
        ----------
        image_zoom     Resolution index = zoom_idx factor, e.g., 0, 1, 2, ...
        ds    HDF5 dataset

        Returns     HDF5 region reference
        -------

        """
        image_fits_header = self.h5_connector.read_serialized_fits_header(image_ds)
        cutout_bounds = get_cutout_bounds(image_fits_header, image_zoom, spec_fits_header, image_cutout_size)
        if not is_cutout_whole(cutout_bounds, image_ds):
            raise NoCoverageFoundError("Cutout not whole.")
        region_ref = self.h5_connector.get_region_ref(image_ds, cutout_bounds)
        return region_ref

    def _get_absolute_heal_paths(self, overlapping_pixel_paths):
        for heal_path in overlapping_pixel_paths:
            absolute_path = "%s/%s" % (self.config.SPARSE_CUBE_NAME, heal_path)
            yield absolute_path


class DatasetSpectrumStrategy(SpectrumMetadataStrategy):

    def __init__(self, metadata_strategy: MetadataStrategy, config: Config, photometry: Photometry, logger):

        super().__init__(metadata_strategy, config, photometry, logger)

        self.index_dtype = [("spatial", np.int64), ("time", np.float32), ("ds_slice_idx", np.int64)]
        self.buffer = {}
        self.metadata_header_buffer = {}
        self.metadata_index_buffer = {}
        self.cutout_data_buffer = None
        self.cutout_error_buffer = None
        self.cutout_metadata_buffer = None
        self.datasets_created = False
        self.target_with_cutout_cnt = 0

    def clear_buffers(self):
        del self.cutout_data_buffer
        del self.cutout_error_buffer
        del self.cutout_metadata_buffer
        del self.buffer
        del self.metadata_index_buffer
        del self.metadata_header_buffer
        self.buffer = {}
        self.metadata_header_buffer = None
        self.metadata_index_buffer = None
        self.cutout_data_buffer = None
        self.cutout_error_buffer = None
        self.cutout_metadata_buffer = None

    def write_datasets(self, res_grp_list, data, file_name, offset, batch_i, batch_size, coordinates=None):
        coordinates = True
        return write_dataset(data, res_grp_list, self.config.FLOAT_COMPRESS, offset, coordinates, self.buffer,
                             batch_size=batch_size, batch_i=batch_i)

    def get_resolution_groups(self, metadata, h5_connector):
        return get_dataset_resolution_groups(h5_connector, self.config.SPARSE_CUBE_NAME, self.config.SPEC_ZOOM_CNT,
                                             "spectra")

    def link_spectra_to_images(self, h5_connector: H5Connector, range_min=None, range_max=None, batch_size=None,
                               image_db_index=None, image_wcs_data=None):
        self.h5_connector = h5_connector 
        self.clear_buffers()  # necessary because linking doesn't always rewrite buffer
        self.spec_cnt = 0
        self.target_with_cutout_cnt = 0
        self.clear_buffers()  # necessary because linking doesn't always rewrite buffer
        total_spectrum_count = h5_connector.get_spectrum_count()
        if not self.config.MPIO:
            spectrum_zoom_groups = require_zoom_grps("spectra", h5_connector, self.config.SPEC_ZOOM_CNT)
            self.recreate_link_datasets(h5_connector, total_spectrum_count, spectrum_zoom_groups)
        if image_db_index is None:
            image_db_index = get_index_datasets(h5_connector, "images", self.config.IMG_ZOOM_CNT,
                                                self.config.SPARSE_CUBE_NAME)[0]
        if image_wcs_data is None:
            image_wcs_data = get_wcs_datasets(h5_connector, "images", self.config.IMG_ZOOM_CNT,
                             self.config.SPARSE_CUBE_NAME)
        image_data_cutout_ds, image_error_cutout_ds, image_metadata_cutout_ds, spectra_metadata_ds = self._get_datasets_for_linking(
            h5_connector)
        if not range_min:
            range_min = 0
        if not range_max:
            range_max = total_spectrum_count
        if not batch_size:
            batch_size = range_max
        spectrum_batch = range(len(range(range_min, range_max)))
        iterator = wrap_tqdm(spectrum_batch, self.config.MPIO, self.__class__.__name__, self.config)
        for i in iterator:
            try:
                self._add_image_links_to_spectra(spectra_metadata_ds, image_data_cutout_ds,
                                                 image_error_cutout_ds, image_metadata_cutout_ds, range_min, i,
                                                 batch_size, image_db_index, image_wcs_data)
            except JSONDecodeError as e:
                self.logger.debug("Could not link images for spectrum %d" % i)
                if self.config.LOG_LEVEL == "DEBUG":
                    traceback.format_exc()
                    raise e
        return self.spec_cnt

    def _get_datasets_for_linking(self, h5_connector):
        spectra_metadata_ds = get_metadata_datasets(h5_connector, "spectra", self.config.SPEC_ZOOM_CNT,
                                                    self.config.SPARSE_CUBE_NAME)
        image_data_cutout_ds = get_cutout_data_datasets(h5_connector, self.config.SPEC_ZOOM_CNT,
                                                        self.config.SPARSE_CUBE_NAME)
        image_error_cutout_ds = get_cutout_error_datasets(h5_connector, self.config.SPEC_ZOOM_CNT,
                                                          self.config.SPARSE_CUBE_NAME)
        image_metadata_cutout_ds = get_cutout_metadata_datasets(h5_connector, self.config.SPEC_ZOOM_CNT,
                                                                self.config.SPARSE_CUBE_NAME)
        return image_data_cutout_ds, image_error_cutout_ds, image_metadata_cutout_ds, spectra_metadata_ds

    def _write_metadata_from_cache(self, h5_connector, fits_headers, no_attrs=False, no_datasets=False, range_min=None,
                                   range_max=None, batch_size=None):
        if not self.config.MPIO:
            self.recreate_datasets(h5_connector)
        super()._write_metadata_from_cache(h5_connector, fits_headers, no_attrs, no_datasets, range_min, range_max,
                                           batch_size)
        if not self.config.MPIO:
            self.sort_indices(h5_connector)

    def recreate_datasets(self, h5_connector):
        self.h5_connector = h5_connector
        spectrum_count = h5_connector.get_spectrum_count()
        spectrum_zoom_groups = require_zoom_grps("spectra", h5_connector, self.config.SPEC_ZOOM_CNT)
        self.create_datasets(spectrum_zoom_groups, spectrum_count)

    def _write_parsed_metadata(self, metadata, fits_path, no_attrs, no_datasets, offset, batch_size):
        spec_datasets = get_data_datasets(self.h5_connector, "spectra", self.config.SPEC_ZOOM_CNT,
                                          self.config.SPARSE_CUBE_NAME)
        fits_name = Path(fits_path).name
        if not no_attrs:
            self.metadata_strategy.add_metadata(self.h5_connector, metadata, spec_datasets, self.spec_cnt, batch_size,
                                                offset, fits_name, self.metadata_header_buffer)
        self.add_index_entry(metadata, spec_datasets, offset, self.spec_cnt, batch_size)

    def create_datasets(self, spec_zoom_groups, spec_count):
        for spec_zoom, spec_zoom_group in enumerate(spec_zoom_groups):
            if "data" not in spec_zoom_group or "errors" not in spec_zoom_group:
                chunk_size = None
                spec_shape = (spec_count, int(self.config.REBIN_SAMPLES / (2 ** spec_zoom)))
                if self.config.DATASET_STRATEGY_CHUNKED:
                    if self.config.SPECTRUM_DATA_BATCH_SIZE > spec_count:
                        chunk_stack_size = 1
                    else:
                        chunk_stack_size = self.config.SPECTRUM_DATA_BATCH_SIZE
                    chunk_size = (chunk_stack_size,) + spec_shape[1:]
                spec_ds = self.h5_connector.create_spectrum_h5_dataset(spec_zoom_group, "data", spec_shape, chunk_size)
                self.h5_connector.set_attr(spec_ds, "mime-type", "spectrum")
                error_ds = self.h5_connector.create_spectrum_h5_dataset(spec_zoom_group, "errors", spec_shape,
                                                                        chunk_size)
                self.h5_connector.create_spectrum_h5_dataset(spec_zoom_group, "wl", (spec_shape[1],))
            else:
                spec_ds = spec_zoom_group["data"]
                error_ds = spec_zoom_group["errors"]
            self.h5_connector.set_attr(spec_ds, "error_ds_ref", error_ds.ref)
            set_nx_axes(spec_zoom_group, [".", "wl"], self.h5_connector)
            set_nx_interpretation(spec_ds, "spectrum", self.h5_connector)
            recreate_additional_datasets(spec_count, spec_ds, spec_zoom_group, self.index_dtype, self.h5_connector,
                                         self.config.FITS_SPECTRUM_MAX_HEADER_SIZE, self.config.FITS_MAX_PATH_SIZE,
                                         self.config.METADATA_CHUNK_SIZE)

    def recreate_link_datasets(self, h5_connector, spec_count, spec_zoom_groups):
        self.h5_connector = h5_connector
        for spec_zoom, spec_zoom_group in enumerate(spec_zoom_groups):
            self.h5_connector.recreate_regionref_dataset("image_cutouts_data", spec_count, spec_zoom_group,
                                                         dtype=self.metadata_strategy.img_region_ref_dtype)
            self.h5_connector.recreate_regionref_dataset("image_cutouts_errors", spec_count, spec_zoom_group,
                                                         dtype=self.metadata_strategy.img_region_ref_dtype)
            self.h5_connector.recreate_regionref_dataset("image_cutouts_metadata", spec_count, spec_zoom_group,
                                                         dtype=self.metadata_strategy.img_region_ref_dtype)

    def add_index_entry(self, metadata, spec_datasets, offset, batch_i, batch_size):
        for zoom_idx, spec_ds in enumerate(spec_datasets):
            spectrum_ra, spectrum_dec = get_spectrum_center_coords(metadata)
            healpix_id = get_healpix_id(spectrum_ra, spectrum_dec, self.config.SPEC_SPAT_INDEX_ORDER - 1)
            time = get_spectrum_time(metadata)
            if zoom_idx not in self.metadata_index_buffer:
                self.metadata_index_buffer[zoom_idx] = np.zeros((batch_size,), self.index_dtype)
            self.metadata_index_buffer[zoom_idx][batch_i] = (healpix_id, time, offset + batch_i)
            if batch_i == (batch_size - 1):
                index_ds = self.h5_connector.file[self.h5_connector.get_attr(spec_ds, "index_ds_ref")]
                index_ds.write_direct(self.metadata_index_buffer[zoom_idx], source_sel=np.s_[0:batch_size],
                                      dest_sel=np.s_[offset:offset + batch_i + 1])

    def sort_indices(self, h5_connector):
        index_datasets = get_index_datasets(h5_connector, "spectra", self.config.SPEC_ZOOM_CNT,
                                            self.config.SPARSE_CUBE_NAME)
        for index_ds in index_datasets:
            index_ds[:] = np.sort(index_ds[:], order=['spatial', 'time'])

    def _add_image_links_to_spectra(self, spec_metadata_datasets_multiple_zoom, image_data_cutout_ds_multiple_zoom,
                                    image_error_cutout_ds_multiple_zoom, image_metadata_cutout_ds_multiple_zoom,
                                    range_min=None, batch_i=None, batch_size=None, image_db_index=None,
                                    image_wcs_data=None):
        """
        Adds HDF5 Region references of image cut-outs to spectra attribute "sparse_cube". Throws NoCoverageFoundError
        if the cut-out does not span the whole cutout path_size for any reason.
        Parameters
        ----------
        spec_metadata_datasets_multiple_zoom   [HDF5 Datasets]

        Returns         [HDF5 Datasets]
        -------

        """
        image_data_datasets, image_error_datasets, image_metadata_datasets = self._get_image_datasets_multiple_zoom()
        image_data_refs = {}
        image_error_refs = {}
        image_metadata_refs = {}
        image_max_zoom = min(self.config.IMG_ZOOM_CNT, self.config.SPEC_ZOOM_CNT)
        original_resolution_dataset = spec_metadata_datasets_multiple_zoom[0]
        spectrum_metadata = self.h5_connector.read_serialized_fits_header(original_resolution_dataset,
                                                                          idx=range_min + self.spec_cnt)
        for image_idx in self._find_images_overlapping_spectrum(spectrum_metadata, image_db_index):
            for image_zoom in range(self.config.IMG_ZOOM_CNT):
                image_data_dataset, image_error_dataset, image_metadata_dataset = self._get_image_ds(
                    image_data_datasets, image_error_datasets, image_metadata_datasets, image_zoom)
                self._write_region_ref_from_image_idx(image_data_refs,
                                                      image_error_refs,
                                                      image_metadata_refs,
                                                      image_data_dataset,
                                                      image_error_dataset,
                                                      image_metadata_dataset,
                                                      image_idx,
                                                      image_zoom,
                                                      spectrum_metadata,
                                                      range_min,
                                                      image_wcs_data[image_zoom])
        self._convert_arrays(image_data_refs, image_error_refs, image_metadata_refs)
        for spec_zoom in range(self.config.SPEC_ZOOM_CNT):
            image_data_dataset, image_error_dataset, image_metadata_dataset = self._get_image_ds(
                image_data_cutout_ds_multiple_zoom, image_error_cutout_ds_multiple_zoom,
                image_metadata_cutout_ds_multiple_zoom, spec_zoom)
            self._write_cutouts(image_data_dataset, image_data_refs, image_error_dataset, image_error_refs,
                                image_max_zoom, image_metadata_dataset, image_metadata_refs, spec_zoom, range_min,
                                batch_i, batch_size)
        self.spec_cnt += 1
        return spec_metadata_datasets_multiple_zoom

    def _write_cutouts(self, image_data_dataset, image_data_refs, image_error_dataset, image_error_refs, image_max_zoom,
                       image_metadata_dataset, image_metadata_refs, spec_zoom, range_min=None, batch_i=None,
                       batch_size=None):
        self._write_image_cutouts(spec_zoom, image_data_dataset, image_max_zoom, image_data_refs, range_min, batch_i,
                                  batch_size, 0)
        self._write_image_cutouts(spec_zoom, image_error_dataset, image_max_zoom, image_error_refs, range_min, batch_i,
                                  batch_size, 1)
        self._write_image_cutouts(spec_zoom, image_metadata_dataset, image_max_zoom, image_metadata_refs, range_min,
                                  batch_i, batch_size, 2)

    def _convert_arrays(self, image_data_refs, image_error_refs, image_metadata_refs):
        self.convert_refs_to_array(image_data_refs, self.metadata_strategy.img_region_ref_dtype)
        self.convert_refs_to_array(image_error_refs, self.metadata_strategy.img_region_ref_dtype)
        self.convert_refs_to_array(image_metadata_refs, self.metadata_strategy.img_region_ref_dtype)

    def _get_image_ds(self, image_data_datasets, image_error_datasets, image_metadata_datasets, image_zoom):
        image_data_dataset = image_data_datasets[image_zoom]
        image_error_dataset = image_error_datasets[image_zoom]
        image_metadata_dataset = image_metadata_datasets[image_zoom]
        return image_data_dataset, image_error_dataset, image_metadata_dataset

    def _get_image_datasets_multiple_zoom(self):
        image_data_datasets = get_data_datasets(self.h5_connector, "images", self.config.SPEC_ZOOM_CNT,
                                                self.config.SPARSE_CUBE_NAME)
        image_error_datasets = get_error_datasets(self.h5_connector, "images", self.config.SPEC_ZOOM_CNT,
                                                  self.config.SPARSE_CUBE_NAME)
        image_metadata_datasets = get_metadata_datasets(self.h5_connector, "images", self.config.SPEC_ZOOM_CNT,
                                                        self.config.SPARSE_CUBE_NAME)
        return image_data_datasets, image_error_datasets, image_metadata_datasets

    def _write_region_ref_from_image_idx(self, image_data_refs, image_error_refs, image_metadata_refs,
                                         image_data_dataset,
                                         image_error_dataset, image_metadata_dataset, image_idx, image_zoom,
                                         spectrum_metadata, offset, image_wcs_data=None):

        if image_zoom not in image_data_refs:
            image_data_refs[image_zoom] = []
        if image_zoom not in image_error_refs:
            image_error_refs[image_zoom] = []
        if image_zoom not in image_metadata_refs:
            image_metadata_refs[image_zoom] = []

        try:
            data_region_ref = self._get_region_ref(image_zoom, image_metadata_dataset, image_data_dataset, image_idx,
                                                   spectrum_metadata,
                                                   self.config.IMAGE_CUTOUT_SIZE, image_wcs_data)
            error_region_ref = self._get_region_ref(image_zoom, image_metadata_dataset, image_error_dataset, image_idx,
                                                    spectrum_metadata,
                                                    self.config.IMAGE_CUTOUT_SIZE, image_wcs_data)
            metadata_region_ref = self.h5_connector.get_metadata_ref(image_metadata_dataset, image_idx)
            image_data_refs[image_zoom].append(data_region_ref)
            image_error_refs[image_zoom].append(error_region_ref)
            image_metadata_refs[image_zoom].append(metadata_region_ref)

        except NoCoverageFoundError as e:
            self.logger.debug("No coverage found for spectrum %s and image %s, reason %s" % (
                    self.spec_cnt + offset, image_idx, str(e)))
        return

    def _write_image_cutout(self, image_cutout_ds, zoom_idx, image_refs, number_of_refs, range_min=None, batch_i=None,
                            batch_size=None, buffer_idx=None):
        if number_of_refs > 0:
            if buffer_idx == 0 and self.cutout_data_buffer is None:
                self.buffer = []
                self.cutout_data_buffer = self.metadata_strategy.get_cutout_buffer(batch_size)
                self.buffer.append(self.cutout_data_buffer)
            elif buffer_idx == 1 and self.cutout_error_buffer is None:
                self.cutout_error_buffer = self.metadata_strategy.get_cutout_buffer(batch_size)
                self.buffer.append(self.cutout_error_buffer)
            elif buffer_idx == 2 and self.cutout_metadata_buffer is None:
                self.cutout_metadata_buffer = self.metadata_strategy.get_cutout_buffer(batch_size)
                self.buffer.append(self.cutout_metadata_buffer)
            self.buffer[buffer_idx][zoom_idx, batch_i, 0:number_of_refs] = image_refs[zoom_idx][:self.config.MAX_CUTOUT_REFS]
            self.target_with_cutout_cnt += 1
        if batch_i == (batch_size - 1) and self.target_with_cutout_cnt > 0:
            image_cutout_ds.write_direct(self.buffer[buffer_idx][zoom_idx], source_sel=np.s_[0:batch_size, ...],
                                         dest_sel=np.s_[range_min:range_min + batch_size, ...])

    def _get_region_ref(self, image_zoom, image_metadata_dataset, image_ds, image_idx, spec_fits_header,
                        image_cutout_size, image_wcs_data=None):
        """
        Gets the region reference for a given resolution from an ds.

        Parameters
        ----------
        image_zoom     Resolution index = zoom_idx factor, e.g., 0, 1, 2, ...
        ds    HDF5 dataset

        Returns     HDF5 region reference
        -------

        """

        if image_wcs_data is not None:
            image_fits_header = image_wcs_data[image_idx]
        else:
            image_fits_header = self.h5_connector.read_serialized_fits_header(image_metadata_dataset, idx=image_idx)
        cutout_bounds = get_cutout_bounds(image_fits_header, image_zoom, spec_fits_header, image_cutout_size)
        if not is_cutout_whole(cutout_bounds, image_ds, image_idx):
            raise NoCoverageFoundError("Cutout not whole.")
        region_ref = self.h5_connector.get_region_ref(image_ds, cutout_bounds, image_idx)
        return region_ref

    def _find_images_overlapping_spectrum(self, metadata, index_dataset_orig_res=None):
        nside = 2 ** (self.config.IMG_SPAT_INDEX_ORDER - 1)
        fact = 2 ** self.config.SPEC_SPAT_INDEX_ORDER
        pix_ids = get_overlapping_healpix_pixel_ids(metadata, nside, fact, self.config.IMG_DIAMETER_ANG_MIN)
        yield from self._get_image_ids_from_pix_ids(pix_ids, index_dataset_orig_res)

    @staticmethod
    def _get_image_ids_from_pix_ids(pix_ids, image_db_index_orig_res):
        for pix_id in pix_ids:
            db_idx = np.searchsorted(image_db_index_orig_res["spatial"], pix_id)
            while db_idx < len(image_db_index_orig_res) and image_db_index_orig_res["spatial"][db_idx] == pix_id:
                img_idx = image_db_index_orig_res[db_idx]["ds_slice_idx"]
                db_idx += 1
                yield img_idx

    @staticmethod
    def convert_refs_to_array(image_refs, dtype=None):
        for res in image_refs:
            image_refs[res] = np.array(image_refs[res], dtype=dtype)


def get_spectrum_time(spec_header):
    try:
        time = spec_header["TAI"]
    except KeyError:
        time = spec_header["MJD"]
    return time
