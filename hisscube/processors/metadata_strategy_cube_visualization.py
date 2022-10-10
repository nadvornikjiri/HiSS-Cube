from abc import ABC, abstractmethod
from pathlib import Path

import h5py
import healpy as hp
import numpy as np
import ujson
from astropy.io.votable import from_table, writeto
from astropy.table import QTable
from tqdm.auto import tqdm

from hisscube.processors.metadata_strategy import dereference_region_ref
from hisscube.processors.metadata_strategy_dataset import DatasetStrategy, get_cutout_data_datasets, \
    get_cutout_error_datasets, get_cutout_metadata_datasets, get_data_datasets, get_error_datasets, get_wl_datasets
from hisscube.processors.metadata_strategy_tree import TreeStrategy
from hisscube.utils.astrometry import get_cutout_pixel_coords
from hisscube.utils.io import get_fits_path, \
    get_spectrum_header_dataset, H5Connector
from hisscube.utils.logging import HiSSCubeLogger
from hisscube.utils.photometry import Photometry


class VisualizationProcessorStrategy(ABC):

    def __init__(self, config):
        self.h5_connector = None
        self.spectrum_metadata = None
        self.config = config
        self.logger = HiSSCubeLogger.logger
        if self.config.INCLUDE_ADDITIONAL_METADATA:  # TODO add the grouprefs to the images and spectra from dense cube
            self.array_type = [('heal_id', '<i8'), ('ra', '<f4'), ('dec', '<f4'), ('time', '<f4'), ('wl', '<f4'),
                               ('mean', '<f4'), ('sigma', '<f4'), ('spec_ra', '<f4'), ('spec_dec', '<f4'),
                               ('fits_name', 'S32'), ('spec_fits_name', 'S32')]
        else:
            self.array_type = [('heal_id', '<i8'), ('ra', '<f8'), ('dec', '<f8'), ('time', '<f8'), ('wl', '<f8'),
                               ('mean', '<f8'), ('sigma', '<f8')]
        self.output_counter = 0
        self.spectral_cube = None
        self.output_zoom = None

    def create_visualization_cube(self, h5_connector: H5Connector):
        self.h5_connector = h5_connector
        self.h5_connector.file = h5_connector.file
        dense_cube_grp = self.h5_connector.file.require_group(self.config.DENSE_CUBE_NAME)
        for zoom in range(
                min(self.config.SPEC_ZOOM_CNT, self.config.IMG_ZOOM_CNT)):
            spectral_cube = self._construct_spectral_cube_table(zoom)
            res_grp = dense_cube_grp.require_group(str(zoom))
            visualization = res_grp.require_group("visualization")
            dense_cube_ds_name = "dense_cube_zoom_%d" % zoom
            if dense_cube_ds_name in visualization:
                del visualization[dense_cube_ds_name]
            ds = h5_connector.create_dataset(visualization, dense_cube_ds_name, spectral_cube.shape,
                                             dataset_type=spectral_cube.dtype)
            ds.write_direct(spectral_cube)

    def read_spectral_cube_table(self, h5_connector, zoom):
        """
        This method just reads the dense cube dataset and returns it as numpy array.
        Parameters
        ----------
        resolution  int

        Returns     numpy array
        -------

        """
        self.h5_connector = h5_connector
        spectral_cube_path = "%s/%d/%s/dense_cube_zoom_%d" % (
            self.config.DENSE_CUBE_NAME, zoom, "visualization", zoom)
        self.spectral_cube = self.h5_connector.file[spectral_cube_path][()]
        return self.spectral_cube

    def _construct_spectral_cube_table(self, zoom):
        """
        This method constructs the dense cube from the semi-sparse cube tree in the HDF5 file for a given resolution.
        It iterates over the individual spectra, reads the "image_cutouts" attribute and de-references all of the
        region references there. Returns coordinates for every voxel of the constructed 4D spectral cube (ra, dec,
        time, wavelength) + optional metadata if enabled by the self.INCLUDE_ADDITIONAL_METADATA.

        Parameters
        ----------
        resolution  int

        Returns
        -------

        """
        self.output_counter = 0
        self.spectral_cube = np.empty((self.config.INIT_ARRAY_SIZE,), dtype=self.array_type)
        self.output_zoom = zoom
        self._construct_multires_spectral_cube_table(self.h5_connector.file)
        truncated_cube = self.spectral_cube[:self.output_counter]
        self.spectral_cube = truncated_cube
        return self.spectral_cube

    def write_VOTable(self, output_path):
        table = self._get_q_table()
        votable = from_table(table)
        writeto(votable, output_path, tabledata_format="binary")

    def write_FITS(self, output_path):
        table = self._get_q_table()
        table.write(output_path, overwrite=True, format='fits')

    def _resize_output_if_necessary(self, spectrum_5d):
        if self.output_counter + spectrum_5d.shape[0] > self.spectral_cube.shape[0]:
            self.spectral_cube.resize((self.spectral_cube.shape[0] * 2,), refcheck=False)

    def _get_table_pixels_from_spectrum_generic(self, dec, ra, res, spectrum_fits_name, spectrum_part):
        heal_id = hp.ang2pix(hp.order2nside(self.config.OUTPUT_HEAL_ORDER),
                             ra, dec,
                             nest=True,
                             lonlat=True)
        time = self.spectrum_metadata["MJD"]
        heal_id_column = np.repeat([heal_id], res, axis=0)
        ra_column = np.repeat([ra], res, axis=0)
        dec_column = np.repeat([dec], res, axis=0)
        time_column = np.repeat([time], res, axis=0)
        if self.config.INCLUDE_ADDITIONAL_METADATA is False:
            spectrum_column_names = 'heal, ra, dec, time, wl, mean, sigma'
            spectrum_columns = [heal_id_column,
                                ra_column,
                                dec_column,
                                time_column,
                                spectrum_part[:, 0].reshape(res, ),
                                spectrum_part[:, 1].reshape(res, ),
                                spectrum_part[:, 2].reshape(res, )]
        else:
            spectrum_column_names = 'heal, ra, dec, time, wl, mean, sigma, spectrum_ra, spectrum_dec, fits_name, ' \
                                    'spectrum_fits_name '
            spectrum_ra_column, spectrum_dec_column, spectrum_name_column = self._get_spectrum_table_columns(res,
                                                                                                             spectrum_fits_name)
            spectrum_columns = [heal_id_column,
                                ra_column,
                                dec_column,
                                time_column,
                                spectrum_part[:, 0].reshape(res, ),
                                spectrum_part[:, 1].reshape(res, ),
                                spectrum_part[:, 2].reshape(res, ),
                                spectrum_ra_column,
                                spectrum_dec_column,
                                spectrum_name_column,
                                spectrum_name_column]
        return np.rec.fromarrays(spectrum_columns, names=spectrum_column_names)

    def _get_table_image_pixels_from_cutout_bounds(self, cutout_bounds, image_path, image_region, spectrum_path, time,
                                                   w,
                                                   wl):
        ra, dec = get_cutout_pixel_coords(cutout_bounds, w)
        no_pixels = ra.size
        spectrum_healpix = hp.ang2pix(hp.order2nside(self.config.OUTPUT_HEAL_ORDER),
                                      self.spectrum_metadata["PLUG_RA"], self.spectrum_metadata["PLUG_DEC"],
                                      nest=True, lonlat=True)
        spec_healpix_column = np.repeat([spectrum_healpix], no_pixels, axis=0)
        ra_column = ra.reshape(no_pixels, )
        dec_column = dec.reshape(no_pixels, )
        data_columns = np.reshape(image_region, (no_pixels, 2))
        wl_column = np.repeat([wl], no_pixels, axis=0)
        time_column = np.repeat([time], no_pixels, axis=0)
        if self.config.INCLUDE_ADDITIONAL_METADATA is False:
            image_column_names = 'heal, ra, dec, time, wl, mean, sigma'
            image_columns = [spec_healpix_column, ra_column, dec_column, time_column, wl_column,
                             data_columns[:, 0].reshape(no_pixels, 1),
                             data_columns[:, 1].reshape(no_pixels, 1)]
        else:
            image_column_names = 'heal, ra, dec, time, wl, mean, sigma, spectrum_ra, spectrum_dec, fits_name, ' \
                                 'spectrum_fits_name '
            image_fits_name = self.parse_str_path(image_path)
            image_fits_name_casted = np.array([image_fits_name]).astype(np.dtype('S32'))
            image_fits_name_column = np.repeat(image_fits_name_casted, no_pixels, axis=0)
            spectrum_fits_name = str(spectrum_path).split('/')[-1]
            spectrum_ra_column, spectrum_dec_column, spectrum_name_column = self._get_spectrum_table_columns(no_pixels,
                                                                                                             spectrum_fits_name)

            image_columns = [spec_healpix_column,
                             ra_column,
                             dec_column,
                             time_column,
                             wl_column,
                             data_columns[:, 0].reshape(no_pixels, ),
                             data_columns[:, 1].reshape(no_pixels, ),
                             spectrum_ra_column,
                             spectrum_dec_column,
                             image_fits_name_column,
                             spectrum_name_column]
        return np.rec.fromarrays(image_columns,
                                 names=image_column_names)

    def parse_str_path(self, image_path):
        return Path(image_path).name

    def _get_spectrum_table_columns(self, no_pixels, spectrum_fits_name):
        spectrum_fits_name_casted = np.array([spectrum_fits_name]).astype(np.dtype('S32'))
        spectrum_ra_column = np.repeat([self.spectrum_metadata["PLUG_RA"]], no_pixels, axis=0)
        spectrum_dec_column = np.repeat([self.spectrum_metadata["PLUG_DEC"]], no_pixels, axis=0)
        spectrum_name_column = np.repeat(spectrum_fits_name_casted, no_pixels, axis=0)
        return spectrum_ra_column, spectrum_dec_column, spectrum_name_column

    def _get_q_table(self):
        if not self.config.INCLUDE_ADDITIONAL_METADATA:
            table = QTable(self.spectral_cube, names=("HealPix ID", "RA", "DEC", "Time", "Wavelength", "Mean", "Sigma"),
                           meta={'name': 'SDSS Cube'})
        else:
            table = QTable(self.spectral_cube, names=("HealPix ID", "RA", "DEC", "Time", "Wavelength", "Mean", "Sigma",
                                                      "Spectrum RA", "Spectrum DEC", "FITS Name", "Spectrum FITS Name"),
                           meta={'name': 'SDSS Cube'})
        return table

    @abstractmethod
    def _construct_multires_spectral_cube_table(self, h5_grp):
        raise NotImplementedError


class TreeVisualizationProcessorStrategy(VisualizationProcessorStrategy):
    def __init__(self, config, metadata_strategy: TreeStrategy):
        super().__init__(config)
        self.metadata_strategy = metadata_strategy

    def _construct_multires_spectral_cube_table(self, h5_parent):
        if "mime-type" in h5_parent.attrs and h5_parent.attrs["mime-type"] == "spectrum":
            if h5_parent.parent.attrs["res_zoom"] == self.output_zoom:
                self._construct_spectrum_table(h5_parent)

        if isinstance(h5_parent, h5py.Group):
            for h5_child in h5_parent.keys():
                self._construct_multires_spectral_cube_table(h5_parent[h5_child])
        return

    def _construct_spectrum_table(self, spectrum_ds):
        """
        Reads the spectral dataset and writes it ot self.spectral_cube.
        Parameters
        ----------
        spectrum_ds HDF5 dataset

        Returns
        -------

        """
        try:
            if spectrum_ds.attrs["orig_res_link"]:
                self.spectrum_metadata = self.h5_connector.read_serialized_fits_header(
                    self.h5_connector.file[spectrum_ds.attrs["orig_res_link"]])
            else:
                self.spectrum_metadata = self.h5_connector.read_serialized_fits_header(spectrum_ds)
        except KeyError:
            self.spectrum_metadata = self.h5_connector.read_serialized_fits_header(spectrum_ds)
        spectrum_5d = self._get_table_pixels_from_spectrum(spectrum_ds.name, spectrum_ds)
        self._resize_output_if_necessary(spectrum_5d)
        self.spectral_cube[self.output_counter:self.output_counter + spectrum_5d.shape[0]] = spectrum_5d
        self.output_counter += spectrum_5d.shape[0]
        cutout_refs = spectrum_ds.parent.parent.parent["image_cutouts_%s" % spectrum_ds.parent.attrs["res_zoom"]]

        if len(cutout_refs) > 0:
            for region_ref in cutout_refs:
                if region_ref:
                    try:
                        image_5d = self._get_table_pixels_from_image_cutout(spectrum_ds, self.output_zoom, region_ref)
                        self._resize_output_if_necessary(image_5d)
                        self.spectral_cube[self.output_counter:self.output_counter + image_5d.shape[0]] = image_5d
                        self.output_counter += image_5d.shape[0]
                    except ValueError as e:
                        self.logger.error("Could not process region for %s, message: %s" % (spectrum_ds.name, str(e)))
                else:
                    break  # necessary because of how null object references are tested in h5py dataset

    def _get_table_pixels_from_spectrum(self, spectrum_h5_path, spectrum_ds):
        """
        Gets array of pixels from the spectrum, also containing their coordinates.
        Parameters
        ----------
        spectrum_h5_path    String
        spectrum_ds         HDF5 dataset

        Returns             numpy record
        -------

        """
        res = int(spectrum_h5_path.split('/')[-2])
        spectrum_fits_name = spectrum_ds.name.split('/')[-1]
        ra, dec = self.spectrum_metadata["PLUG_RA"], self.spectrum_metadata["PLUG_DEC"]
        spectrum_part = spectrum_ds
        return self._get_table_pixels_from_spectrum_generic(dec, ra, res, spectrum_fits_name, spectrum_part)

    def _get_table_pixels_from_image_cutout(self, spectrum_ds, res_idx, region_ref):
        """
        Gets all of the image pixels for the given cutout, along with its coordinates.
        Parameters
        ----------
        spectrum_ds HDF5 dataset
        image_zoom     int
        region_ref  HDF5 region reference

        Returns     numpy record
        -------

        """
        spectrum_path = spectrum_ds.name
        image_ds = self.h5_connector.file[region_ref]
        image_path = image_ds.name
        image_region = image_ds[region_ref]

        cutout_bounds, time, w, wl = self.metadata_strategy.get_cutout_bounds_from_spectrum(self.h5_connector, image_ds,
                                                                                            spectrum_ds,
                                                                                            self.config.IMAGE_CUTOUT_SIZE,
                                                                                            res_idx)
        return self._get_table_image_pixels_from_cutout_bounds(cutout_bounds, image_path, image_region, spectrum_path,
                                                               time, w,
                                                               wl)


class DatasetVisualizationProcessorStrategy(VisualizationProcessorStrategy):

    def __init__(self, config, photometry: Photometry, metadata_strategy: DatasetStrategy):
        super().__init__(config)
        self.photometry = photometry
        self.metadata_strategy = metadata_strategy
        self.spec_cnt = 0

    def _construct_multires_spectral_cube_table(self, h5_parent):
        spec_data_dataset_multiple_zoom = get_data_datasets(self.h5_connector, "spectra", self.config.SPEC_ZOOM_CNT,
                                                            self.config.ORIG_CUBE_NAME)
        spec_data_ds = spec_data_dataset_multiple_zoom[self.output_zoom]
        spec_error_dataset_multiple_zoom = get_error_datasets(self.h5_connector, "spectra", self.config.SPEC_ZOOM_CNT,
                                                              self.config.ORIG_CUBE_NAME)
        spec_error_ds = spec_error_dataset_multiple_zoom[self.output_zoom]
        spec_wl_dataset_multiple_zoom = get_wl_datasets(self.h5_connector, "spectra", self.config.SPEC_ZOOM_CNT,
                                                        self.config.ORIG_CUBE_NAME)
        spec_wl_ds = spec_wl_dataset_multiple_zoom[self.output_zoom]
        spec_cnt_total = self.h5_connector.get_spectrum_count()
        spec_wl_for_all_spectra = np.repeat([spec_wl_ds], spec_cnt_total, axis=0)
        spec_ds = np.dstack([spec_wl_for_all_spectra, spec_data_ds, spec_error_ds])
        cutout_data_datasets_multiple_zoom = get_cutout_data_datasets(self.h5_connector, self.config.SPEC_ZOOM_CNT,
                                                                      self.config.ORIG_CUBE_NAME)
        cutout_error_datasets_multiple_zoom = get_cutout_error_datasets(self.h5_connector, self.config.SPEC_ZOOM_CNT,
                                                                        self.config.ORIG_CUBE_NAME)
        cutout_metadata_datasets_multiple_zoom = get_cutout_metadata_datasets(self.h5_connector,
                                                                              self.config.SPEC_ZOOM_CNT,
                                                                              self.config.ORIG_CUBE_NAME)
        cutout_data_refs = cutout_data_datasets_multiple_zoom[self.output_zoom]
        cutout_error_refs = cutout_error_datasets_multiple_zoom[self.output_zoom]
        cutout_metadata_refs = cutout_metadata_datasets_multiple_zoom[self.output_zoom]

        for spec_idx in tqdm(range(spec_cnt_total),
                             desc="Building zoom %d" % self.output_zoom, position=0, leave=True):
            self._construct_spectrum_table(spec_ds, spec_idx, cutout_data_refs, cutout_error_refs, cutout_metadata_refs)

    def parse_str_path(self, image_path):
        image_path = image_path.decode('utf-8')
        return super().parse_str_path(image_path)

    def _construct_spectrum_table(self, spectra_ds, spec_idx, cutout_data_refs, cutout_error_refs,
                                  cutout_metadata_refs):
        """
        Reads the spectral dataset and writes it ot self.spectral_cube.
        Parameters
        ----------
        spectrum_ds HDF5 dataset

        Returns
        -------

        """
        orig_spectrum_fits_header_dataset = get_spectrum_header_dataset(self.h5_connector)
        spectrum_fits_path = get_fits_path(orig_spectrum_fits_header_dataset, spec_idx)
        metadata_orig_zoom = self.h5_connector.read_serialized_fits_header(orig_spectrum_fits_header_dataset,
                                                                           idx=spec_idx)
        self.spectrum_metadata = metadata_orig_zoom
        spectrum_ds = spectra_ds[spec_idx]
        spectrum_5d = self._get_table_pixels_from_spectrum(spectrum_fits_path, spectrum_ds)
        self._resize_output_if_necessary(spectrum_5d)
        self.spectral_cube[self.output_counter:self.output_counter + spectrum_5d.shape[0]] = spectrum_5d
        self.output_counter += spectrum_5d.shape[0]

        for i in range(self.config.MAX_CUTOUT_REFS):
            cutout_data_ref = cutout_data_refs[spec_idx][i]
            cutout_error_ref = cutout_error_refs[spec_idx][i]
            cutout_metadata_ref = cutout_metadata_refs[spec_idx][i]
            if cutout_data_ref and cutout_error_ref and cutout_metadata_ref:
                try:
                    image_5d = self._get_table_pixels_from_image_cutout(spectrum_ds, spectrum_fits_path,
                                                                        self.output_zoom, cutout_data_ref,
                                                                        cutout_error_ref, cutout_metadata_ref)
                    self._resize_output_if_necessary(image_5d)
                    self.spectral_cube[self.output_counter:self.output_counter + image_5d.shape[0]] = image_5d
                    self.output_counter += image_5d.shape[0]
                except ValueError as e:
                    self.logger.error("Could not process region for %s, message: %s" % (spectrum_ds, str(e)))
            else:
                break  # necessary because of how null object references are tested in h5py dataset

    def _get_table_pixels_from_spectrum(self, spectrum_fits_path, spectrum_ds):
        """
        Gets array of pixels from the spectrum, also containing their coordinates.
        Parameters
        ----------
        spectrum_h5_path    String
        spectrum_ds         HDF5 dataset

        Returns             numpy record
        -------

        """
        res = int(self.config.REBIN_SAMPLES / (2 ** self.output_zoom))
        spectrum_fits_name = Path(spectrum_fits_path.decode('utf-8')).name
        ra, dec = self.spectrum_metadata["PLUG_RA"], self.spectrum_metadata["PLUG_DEC"]
        spectrum_part = spectrum_ds
        return self._get_table_pixels_from_spectrum_generic(dec, ra, res, spectrum_fits_name, spectrum_part)

    def _get_table_pixels_from_image_cutout(self, spectrum_ds, spectrum_fits_path, res_idx, region_data_ref,
                                            region_error_ref, cutout_metadata_ref):
        """
        Gets all of the image pixels for the given cutout, along with its coordinates.
        Parameters
        ----------
        spectrum_ds HDF5 dataset
        image_zoom     int
        region_ref  HDF5 region reference

        Returns     numpy record
        -------

        """
        cutout_metadata = dereference_region_ref(cutout_metadata_ref, self.h5_connector)
        image_path = cutout_metadata["path"]
        image_fits_header = ujson.loads(cutout_metadata["header"])
        image_data_region = dereference_region_ref(region_data_ref, self.h5_connector)
        image_error_region = dereference_region_ref(region_error_ref, self.h5_connector)
        image_region = np.dstack([image_data_region, image_error_region])
        cutout_bounds, time, w, wl = self.metadata_strategy.get_cutout_bounds_from_spectrum(image_fits_header, res_idx,
                                                                                            self.spectrum_metadata,
                                                                                            self.photometry)
        return self._get_table_image_pixels_from_cutout_bounds(cutout_bounds, image_path, image_region,
                                                               spectrum_fits_path,
                                                               time, w,
                                                               wl)
