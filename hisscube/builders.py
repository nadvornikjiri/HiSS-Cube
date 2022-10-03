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
from hisscube.utils.io import truncate, H5Connector, CBoostedMetadataBuildWriter, get_image_header_dataset, \
    get_spectrum_header_dataset, get_str_paths, get_image_str_paths, get_spectra_str_paths
from hisscube.utils.logging import HiSSCubeLogger
from hisscube.utils.mpi_helper import chunks, MPIHelper
from hisscube.utils.photometry import Photometry


class HiSSCubeConstructionDirector:
    def __init__(self, cli_args, config: Config, serial_builders,
                 parallel_builders):
        self.args = cli_args
        self.config = config
        self.serial_builders = serial_builders
        self.parallel_builders = parallel_builders
        self.h5_path = cli_args.output_path
        self.builders = []

    def construct(self):

        if self.args.command == "create":
            truncate(self.h5_path)
            self.builders.append(self.serial_builders.metadata_cache_builder)
            self.append_metadata_builder()
            self.append_data_builder()
            if self.config.CREATE_REFERENCES:
                self.builders.append(self.serial_builders.link_builder)
            if self.config.CREATE_VISUALIZATION_CUBE:
                self.builders.append(self.serial_builders.visualization_cube_builder)
            if self.config.CREATE_ML_CUBE:
                self.builders.append(self.serial_builders.ml_cube_builder)

        elif self.args.command == "update":
            if self.args.truncate:
                truncate(self.h5_path)
            if self.args.fits_metadata_cache:
                self.builders.append(self.serial_builders.metadata_cache_builder)
            if self.args.spectrum_metadata:
                self.append_metadata_builder()
            if self.args.data:
                self.append_data_builder()
            if self.args.link_images_spectra:
                self.builders.append(self.serial_builders.link_builder)
            if self.args.visualization_cube:
                self.builders.append(self.serial_builders.visualization_cube_builder)
            if self.args.ml_cube:
                self.builders.append(self.serial_builders.ml_cube_builder)

        for builder in self.builders:
            builder.build()

    def append_metadata_builder(self):
        if self.config.C_BOOSTER and self.config.METADATA_STRATEGY == "TREE":
            self.builders.append(self.serial_builders.c_boosted_metadata_builder)
        else:
            self.builders.append(self.serial_builders.metadata_builder)

    def append_data_builder(self):
        if self.config.MPIO:
            if self.config.PARALLEL_MODE == "MWMR":
                self.builders.append(self.parallel_builders.data_builder_MWMR)
            elif self.config.PARALLEL_MODE == "SMWR":
                self.builders.append(self.parallel_builders.data_builder_SWMR)
        else:
            self.builders.append(self.serial_builders.data_builder)


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
            self.logger.info("Writing image spectrum_metadata.")
            self.image_processor.write_images_metadata(h5_connector)
            self.logger.info("Writing spectra spectrum_metadata.")
            self.spectrum_processor.write_spectra_metadata(h5_connector)


class CBoosterMetadataBuilder(MetadataBuilder):
    def _build(self):
        with self.h5_connector as h5_connector:
            self.logger.info("Creating image spectrum_metadata in memory.")
            self.image_processor.write_images_metadata(h5_connector)
            self.logger.info("Creating spectra spectrum_metadata in memory.")
            self.spectrum_processor.write_spectra_metadata(h5_connector)
            self.c_write_hdf5_metadata(h5_connector)

    def c_write_hdf5_metadata(self, h5_connector: CBoostedMetadataBuildWriter):
        self.logger.info("Initiating C booster for spectrum_metadata write.")
        if self.config.CHUNK_SIZE:
            chunk_size = literal_eval(self.config.CHUNK_SIZE)
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


class ParallelDataBuilder(Builder):
    def __init__(self, config: Config, h5_connector: H5Connector, mpi_helper: MPIHelper,
                 metadata_processor: MetadataProcessor, image_processor: ImageProcessor,
                 spectrum_processor: SpectrumProcessor, photometry: Photometry):
        super().__init__(config, h5_connector)
        self.mpi_helper = mpi_helper
        self.metadata_processor = metadata_processor
        self.image_processor = image_processor
        self.spectrum_processor = spectrum_processor
        self.photometry = photometry

    def build(self):
        self.mpi_helper.barrier()
        with self.h5_connector as h5_connector:
            if self.rank == 0:
                image_paths = get_str_paths(get_image_header_dataset(h5_connector))
                self.distribute_work(h5_connector, image_paths,
                                     self.image_processor)
            else:
                self.process_image_data(h5_connector)
            self.mpi_helper.barrier()
            if self.mpi_helper.rank == 0:
                spectra_paths = get_str_paths(get_spectrum_header_dataset(h5_connector))
                self.distribute_work(h5_connector, spectra_paths, self.spectrum_processor)
            else:
                self.process_spectra_data(h5_connector)
            self.mpi_helper.barrier()

    def distribute_work(self, h5_connector, path_list, processor):
        status = self.mpi_helper.MPI.Status()
        path_list = list(path_list)
        batches = list(chunks(path_list, self.config.BATCH_SIZE))
        offset = 0
        for i in tqdm(range(1, len(batches) + 1)):
            next_batch_size = len(batches[0])
            if i < self.mpi_helper.size:
                self.mpi_helper.send_work(batches, dest=i, offset=offset)
                self.mpi_helper.active_workers += 1
            else:
                self.mpi_helper.wait_for_message(source=self.mpi_helper.MPI.ANY_SOURCE,
                                                 tag=self.mpi_helper.FINISHED_TAG, status=status)
                self.process_response(h5_connector, processor, status)
                self.mpi_helper.send_work(batches, status.Get_source(), offset=offset)
            offset += next_batch_size
        for i in range(1, self.mpi_helper.size):
            self.mpi_helper.send_work_finished(dest=i)
        for i in range(0, self.mpi_helper.active_workers):
            self.process_response(h5_connector, processor, status)
        self.mpi_helper.active_workers = 0

    @abstractmethod
    def process_image_data(self, h5_connector):
        pass

    @abstractmethod
    def process_spectra_data(self, h5_connector):
        pass

    @abstractmethod
    def process_response(self, h5_connector, processor, status):
        pass


class ParallelMWMRDataBuilder(ParallelDataBuilder):
    def __init__(self, config: Config, h5_connector: H5Connector, mpi_helper: MPIHelper,
                 metadata_processor: MetadataProcessor, image_processor: ImageProcessor,
                 spectrum_processor: SpectrumProcessor, photometry: Photometry):
        super().__init__(config, h5_connector, mpi_helper, metadata_processor, image_processor,
                         spectrum_processor, photometry)

    @staticmethod
    def write_data(metadata, data, h5_connector, fits_path, processor, offset):
        res_grps = processor.get_resolution_groups(metadata, h5_connector)
        file_name = Path(fits_path).name
        processor.write_datasets(res_grps, data, file_name, offset=offset)

    def process_image_data(self, h5_connector):
        status = self.mpi_helper.MPI.Status()
        image_path_list, offset = self.mpi_helper.receive_work_parsed(status)

        while status.Get_tag() != self.mpi_helper.KILL_TAG:
            self.logger.debug("Status tag: %d" % status.Get_tag())
            for i, image_path in enumerate(image_path_list):
                try:
                    self.logger.debug("Rank %02d: Processing image %s." % (self.mpi_helper.rank, image_path))
                    metadata, data = self.photometry.get_multiple_resolution_image(image_path, self.config.IMG_ZOOM_CNT)
                    self.write_data(metadata, data, h5_connector, image_path, self.image_processor, offset + i)
                except Exception as e:
                    self.logger.warning("Could not process image %s, message: %s" % (image_path, str(e)))
            self.mpi_helper.comm.send(obj=None, tag=self.mpi_helper.FINISHED_TAG, dest=0)
            image_path_list, offset = self.mpi_helper.receive_work_parsed(status)

    def process_spectra_data(self, h5_connector):
        status = self.mpi_helper.MPI.Status()
        spectra_path_list, offset = self.mpi_helper.receive_work_parsed(status)
        while status.Get_tag() != self.mpi_helper.KILL_TAG:
            for i, spec_path in enumerate(spectra_path_list):
                try:
                    self.logger.debug("Rank %02d: Processing spectrum %s." % (self.mpi_helper.rank, spec_path))
                    metadata, data = self.photometry.get_multiple_resolution_spectrum(
                        spec_path, self.config.SPEC_ZOOM_CNT,
                        apply_rebin=self.config.APPLY_REBIN,
                        rebin_min=self.config.REBIN_MIN,
                        rebin_max=self.config.REBIN_MAX,
                        rebin_samples=self.config.REBIN_SAMPLES,
                        apply_transmission=self.config.APPLY_TRANSMISSION_CURVE)
                    self.write_data(metadata, data, h5_connector, spec_path, self.spectrum_processor, offset + i)
                except Exception as e:
                    self.logger.warning("Could not process spectrum %s, message: %s" % (spec_path, str(e)))
            self.mpi_helper.comm.send(obj=None, tag=self.mpi_helper.FINISHED_TAG, dest=0)
            spectra_path_list, offset = self.mpi_helper.receive_work_parsed(status)

    def process_response(self, h5_connector, processor, status):
        self.mpi_helper.comm.recv(source=self.mpi_helper.MPI.ANY_SOURCE, tag=self.mpi_helper.FINISHED_TAG,
                                  status=status)
        self.h5_connector.fits_total_cnt += 1
        self.logger.debug(
            "Received response from. dest %02d: %d " % (status.Get_source(), self.mpi_helper.sent_work_cnt))


class ParallelSWMRDataBuilder(ParallelDataBuilder):
    def __init__(self, config: Config, h5_connector: H5Connector, mpi_helper: MPIHelper,
                 metadata_processor: MetadataProcessor, image_processor: ImageProcessor,
                 spectrum_processor: SpectrumProcessor, photometry: Photometry):
        super().__init__(config, h5_connector, mpi_helper, metadata_processor, image_processor,
                         spectrum_processor, photometry)
        self.comm_buffer = bytearray(
            self.config.BATCH_SIZE * 100 * 1024 * 1024)  # 100 MBs for one image

    def process_image_data(self, h5_connector):
        status = self.mpi_helper.MPI.Status()
        image_path_list = self.mpi_helper.receive_work(status)

        while status.Get_tag() != self.mpi_helper.KILL_TAG:
            processed_image_batch = []
            for image_path in image_path_list:
                self.logger.info("Rank %02d: Processing image %s." % (self.mpi_helper.rank, image_path))
                metadata, data = self.photometry.get_multiple_resolution_image(image_path, self.config.IMG_ZOOM_CNT)
                self.add_to_result_batch(metadata, data, image_path, processed_image_batch)
            self.mpi_helper.comm.send(obj=processed_image_batch, dest=0)
            image_path_list = self.mpi_helper.receive_work(status)

    def process_spectra_data(self, h5_connector):
        status = self.mpi_helper.MPI.Status()
        spectra_path_list = self.mpi_helper.receive_work(status)
        while status.Get_tag() != self.mpi_helper.KILL_TAG:
            processed_spectra_batch = []
            for spec_path in spectra_path_list:
                self.logger.info("Rank %02d: Processing spectrum %s." % (self.mpi_helper.rank, spec_path))
                metadata, data = self.photometry.get_multiple_resolution_spectrum(
                    spec_path, self.config.SPEC_ZOOM_CNT,
                    apply_rebin=self.config.APPLY_REBIN,
                    rebin_min=self.config.REBIN_MIN,
                    rebin_max=self.config.REBIN_MAX,
                    rebin_samples=self.config.REBIN_SAMPLES,
                    apply_transmission=self.config.APPLY_TRANSMISSION_CURVE)
                self.add_to_result_batch(metadata, data, spec_path, processed_spectra_batch)
            self.mpi_helper.comm.send(obj=processed_spectra_batch, dest=0)
            spectra_path_list = self.mpi_helper.receive_work(status)

    def process_response(self, h5_connector, processor, status):
        for response in self.comm_buffer:
            metadata = response["spectrum_metadata"]
            data = response["data"]
            file_name = response["file_name"]
            res_grps = processor.get_resolution_groups(metadata, h5_connector)
            processor.write_datasets(res_grps, data, file_name)

    @staticmethod
    def add_to_result_batch(metadata, data, fits_path, processed_image_batch):
        file_name = Path(fits_path).name
        if "COMMENT" in metadata:
            del metadata["COMMENT"]  # TODO fix this serialization hack.
        if "HISTORY" in metadata:
            del metadata["HISTORY"]
        processed_image_batch.append(
            {"spectrum_metadata": dict(metadata), "data": data,
             "file_name": str(file_name)})


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
            self.ml_processor.h5_connector = h5_connector
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
