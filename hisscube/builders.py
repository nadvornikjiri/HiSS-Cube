import pathlib
import string
from abc import ABC, abstractmethod
from ast import literal_eval
from pathlib import Path
from tqdm import tqdm
import h5py
from h5writer import write_hdf5_metadata

from hisscube.processors.cube_ml import MLProcessor
from hisscube.processors.cube_visualization import VisualizationProcessor
from hisscube.processors.metadata import MetadataProcessor
from hisscube.processors.image import ImageProcessor
from hisscube.processors.spectrum import SpectrumProcessor
from hisscube.utils.config import Config
from hisscube.utils.io import H5Connector, CBoostedMetadataBuildWriter, get_image_str_paths, get_spectra_str_paths
from hisscube.utils.logging import HiSSCubeLogger
from hisscube.utils.mpi_helper import MPIHelper
from hisscube.utils.photometry import Photometry


class Builder(ABC):
    def __init__(self, config: Config, h5_connector: H5Connector):
        self.rank = MPIHelper.rank
        self.logger = HiSSCubeLogger.logger
        self.config = config
        self.h5_connector = h5_connector

    @abstractmethod
    def build(self):
        raise NotImplementedError


class SingleImageBuilder(Builder):
    def __init__(self, config: Config, h5_connector: H5Connector,
                 image_processor: ImageProcessor,
                 photometry: Photometry, image_path=None):
        super().__init__(config, h5_connector)
        self.image_path = image_path
        self.image_processor = image_processor
        self.photometry = photometry

    def build(self):
        if not self.image_path:
            raise RuntimeError("You need to set the fits_name first.")
        with self.h5_connector as h5_connector:
            return self.build_image(h5_connector, self.image_path, self.image_processor)

    def build_image(self, h5_connector, image_path, processor):
        """
        Method that writes an image to the opened HDF5 file (self.file).
        Parameters
        ----------
        fits_name  String

        Returns     HDF5 Dataset (already written to the file)
        -------

        """
        path = Path(image_path)
        fits_folder_path = path.parent
        fits_file_name = path.name
        processor.update_fits_metadata_cache(h5_connector, fits_folder_path, image_pattern=fits_file_name)
        processor.write_images_metadata(h5_connector)
        img_datasets = self.build_data(h5_connector, image_path, processor, fits_file_name)
        return img_datasets

    def build_data(self, h5_connector, image_path, image_metadata_processor, fits_file_name, offset=0):
        metadata, data = self.photometry.get_multiple_resolution_image(image_path, self.config.IMG_ZOOM_CNT)
        res_grp_list = image_metadata_processor.get_resolution_groups(metadata, h5_connector)
        img_datasets = image_metadata_processor.write_datasets(res_grp_list, data, fits_file_name, offset)
        return img_datasets


class SingleSpectrumBuilder(Builder):
    def __init__(self, config: Config, h5_connector: H5Connector,
                 spectrum_processor: SpectrumProcessor,
                 photometry: Photometry, spectrum_path=None):
        super().__init__(config, h5_connector)
        self.spectrum_path = spectrum_path
        self.spectrum_processor = spectrum_processor
        self.photometry = photometry

    def build(self):
        if not self.spectrum_path:
            raise RuntimeError("You need to set the spectrum_path first.")
        with self.h5_connector as h5_connector:
            return self.build_spectrum(h5_connector, self.spectrum_path, self.spectrum_processor)

    def build_spectrum(self, h5_connector, spec_path, processor):
        """
        Method that writes a spectrum to the opened HDF5 file (self.file). Needs to be called after all images are already
        ingested, as it also links the spectra to the images via the Region References.
        Parameters
        ----------
        spec_path   String

        Returns     HDF5 dataset (already written to the file)
        -------

        """
        path = pathlib.Path(spec_path)
        fits_folder_path = path.parent
        fits_file_name = path.name
        processor.update_fits_metadata_cache(h5_connector, fits_folder_path,
                                             spec_pattern=fits_file_name)
        processor.write_spectra_metadata(h5_connector)
        spec_datasets = self.build_data(h5_connector, spec_path, processor, fits_file_name)
        return spec_datasets

    def build_data(self, h5_connector, spec_path, spectrum_processor, fits_file_name, offset=0):
        metadata, data = self.photometry.get_multiple_resolution_spectrum(
            spec_path, self.config.SPEC_ZOOM_CNT,
            apply_rebin=self.config.APPLY_REBIN,
            rebin_min=self.config.REBIN_MIN,
            rebin_max=self.config.REBIN_MAX,
            rebin_samples=self.config.REBIN_SAMPLES,
            apply_transmission=self.config.APPLY_TRANSMISSION_CURVE)
        res_grp_list = spectrum_processor.get_resolution_groups(metadata, h5_connector)
        spec_datasets = spectrum_processor.write_datasets(res_grp_list, data, fits_file_name, offset)
        return spec_datasets


class SerialBuilder(Builder):

    def build(self):
        if self.rank == 0:
            self._build()

    @abstractmethod
    def _build(self):
        raise NotImplementedError


class MetadataCacheBuilder(SerialBuilder):
    def __init__(self, fits_image_path: string, fits_spectra_path: string, config: Config,
                 h5_connector: H5Connector, metadata_processor: MetadataProcessor, fits_image_pattern=None,
                 fits_spectra_pattern=None):
        super().__init__(config, h5_connector)
        self.fits_image_path = fits_image_path
        self.fits_spectra_path = fits_spectra_path
        self.metadata_processor = metadata_processor
        self.fits_image_pattern = fits_image_pattern
        self.fits_spectra_pattern = fits_spectra_pattern

    def _build(self):
        with self.h5_connector as h5_connector:
            self.logger.info("Writing fits header cache.")
            self.metadata_processor.reingest_fits_tables(h5_connector, self.fits_image_path, self.fits_spectra_path,
                                                         image_pattern=self.fits_image_pattern,
                                                         spectra_pattern=self.fits_spectra_pattern)


class MetadataBuilder(SerialBuilder):
    def __init__(self, config: Config, h5_connector: H5Connector,
                 image_processor: ImageProcessor,
                 spectrum_processor: SpectrumProcessor):
        super().__init__(config, h5_connector)
        self.image_processor = image_processor
        self.spectrum_processor = spectrum_processor

    def _build(self):
        with self.h5_connector as h5_connector:
            self.logger.info("Writing image metadata.")
            self.image_processor.write_images_metadata(h5_connector)
            self.logger.info("Writing spectra metadata.")
            self.spectrum_processor.write_spectra_metadata(h5_connector)


class CBoosterMetadataBuilder(MetadataBuilder):
    def _build(self):
        with self.h5_connector as h5_connector:
            self.logger.info("Creating image metadata in memory.")
            self.image_processor.write_images_metadata(h5_connector)
            self.logger.info("Creating spectra metadata in memory.")
            self.spectrum_processor.write_spectra_metadata(h5_connector)
            self.c_write_hdf5_metadata(h5_connector)

    def c_write_hdf5_metadata(self, h5_connector: CBoostedMetadataBuildWriter):
        self.logger.info("Initiating C booster for metadata write.")
        if self.config.IMAGE_CHUNK_SIZE:
            chunk_size = literal_eval(self.config.IMAGE_CHUNK_SIZE)
            write_hdf5_metadata(h5_connector.h5_file_structure, h5_connector.h5_path, h5_connector.c_timing_log, 1,
                                chunk_size[0],
                                chunk_size[1], chunk_size[2])
        else:
            write_hdf5_metadata(h5_connector.h5_file_structure, h5_connector.h5_path, h5_connector.c_timing_log, 0, 0,
                                0, 0)


class DataBuilder(SerialBuilder):
    def __init__(self, config: Config, h5_connector: H5Connector,
                 image_processor: ImageProcessor,
                 spectrum_processor: SpectrumProcessor,
                 single_image_builder: SingleImageBuilder,
                 single_spectrum_builder: SingleSpectrumBuilder):
        super().__init__(config, h5_connector)
        self.image_processor = image_processor
        self.spectrum_processor = spectrum_processor
        self.single_image_builder = single_image_builder
        self.single_spectrum_builder = single_spectrum_builder

    def _build(self):
        with self.h5_connector as h5_connector:
            image_path_list = get_image_str_paths(h5_connector)
            spec_path_list = get_spectra_str_paths(h5_connector)
            for image_offset, image_path in enumerate(tqdm(image_path_list, desc="Image Data Progress: ")):
                fits_file_name = Path(image_path).name
                self.single_image_builder.build_data(h5_connector, image_path, self.image_processor,
                                                     fits_file_name, offset=image_offset)
            for spectrum_offset, spec_path in enumerate(tqdm(spec_path_list, desc="Spectra Data Progress: ")):
                fits_file_name = Path(spec_path).name
                self.single_spectrum_builder.build_data(h5_connector, spec_path, self.spectrum_processor,
                                                        fits_file_name, offset=spectrum_offset)


class LinkBuilder(SerialBuilder):
    def __init__(self, config: Config, h5_connector: H5Connector,
                 spectrum_processor: SpectrumProcessor):
        super().__init__(config, h5_connector)
        self.spectrum_processor = spectrum_processor

    def _build(self):
        with self.h5_connector as h5_connector:
            self.logger.info("Creating region references to link spectra to image cutouts.")
            self.spectrum_processor.link_spectra_to_images(h5_connector)


class MLCubeBuilder(SerialBuilder):
    def __init__(self, config: Config, h5_connector: H5Connector, ml_processor: MLProcessor):
        super().__init__(config, h5_connector)
        self.ml_processor = ml_processor

    def _build(self):
        with self.h5_connector as h5_connector:
            self.logger.info("Creating the ML 3D dense cube.")
            self.ml_processor.create_3d_cube(h5_connector)


class VisualizationCubeBuilder(SerialBuilder):
    def __init__(self, config: Config, h5_connector: H5Connector,
                 visualization_processor: VisualizationProcessor):
        super().__init__(config, h5_connector)
        self.visualization_processor = visualization_processor

    def _build(self):
        """
        Creates the dense cube Group and datasets, needs to be called after the the images and spectra were already
        ingested.
        Returns
        -------

        """
        with self.h5_connector as h5_connector:
            self.logger.info("Creating the visualization dense cube.")
            self.visualization_processor.create_visualization_cube(h5_connector)
