import string
from abc import ABCMeta, abstractmethod
from pathlib import Path

from tqdm import tqdm

from hisscube.builders import Builder
from hisscube.processors.image import ImageProcessor
from hisscube.processors.metadata import MetadataProcessor
from hisscube.processors.spectrum import SpectrumProcessor
from hisscube.utils.config import Config
from hisscube.utils.io import H5Connector, get_str_paths, get_image_header_dataset, get_spectrum_header_dataset, \
    SerialH5Writer, ParallelH5Writer
from hisscube.utils.mpi_helper import MPIHelper, chunks
from hisscube.utils.photometry import Photometry


class ParallelBuilder(Builder, metaclass=ABCMeta):

    def __init__(self, config: Config, h5_connector: H5Connector, mpi_helper: MPIHelper):
        super().__init__(config, h5_connector)
        self.mpi_helper = mpi_helper

    def distribute_work(self, h5_connector, path_list, processor):
        status = self.mpi_helper.MPI.Status()
        path_list = list(path_list)
        batches = list(chunks(path_list, self.config.BATCH_SIZE))
        offset = 0
        for i in tqdm(range(1, len(batches) + 1), desc=("%s progress" % self.__class__.__name__)):
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

    def process_response(self, h5_connector, processor, status):
        self.mpi_helper.comm.recv(source=self.mpi_helper.MPI.ANY_SOURCE, tag=self.mpi_helper.FINISHED_TAG,
                                  status=status)
        self.h5_connector.fits_total_cnt += 1
        self.logger.debug(
            "Received response from. dest %02d: %d " % (status.Get_source(), self.mpi_helper.sent_work_cnt))

    def process_path(self, h5_connector, process_func, *args, **kwargs):
        status = self.mpi_helper.MPI.Status()
        path_list, offset = self.mpi_helper.receive_work_parsed(status)
        while status.Get_tag() != self.mpi_helper.KILL_TAG:
            for i, path in enumerate(path_list):
                try:
                    process_func(h5_connector, i, path, offset, *args, **kwargs)
                except Exception as e:
                    self.logger.warning("Could not process %s, message: %s" % (path, str(e)))
            self.mpi_helper.comm.send(obj=None, tag=self.mpi_helper.FINISHED_TAG, dest=0)
            path_list, offset = self.mpi_helper.receive_work_parsed(status)


class ParallelMetadataCacheBuilder(ParallelBuilder):
    def __init__(self, fits_image_path: string, fits_spectra_path: string, config: Config,
                 serial_h5_connector: SerialH5Writer, parallel_h5_connector: ParallelH5Writer, mpi_helper: MPIHelper,
                 metadata_processor: MetadataProcessor,
                 fits_image_pattern=None,
                 fits_spectra_pattern=None):
        super().__init__(config, serial_h5_connector, mpi_helper)
        self.fits_image_path = fits_image_path
        self.fits_spectra_path = fits_spectra_path
        self.metadata_processor = metadata_processor
        self.fits_image_pattern = fits_image_pattern
        self.fits_spectra_pattern = fits_spectra_pattern
        self.parallel_connector = parallel_h5_connector

    def build(self):
        if self.rank == 0:
            with self.h5_connector as h5_connector:
                image_path_list, spectra_path_list = self.metadata_processor.parse_paths(self.fits_image_path,
                                                                                         self.fits_image_pattern,
                                                                                         self.fits_spectra_path,
                                                                                         self.fits_spectra_pattern)
                image_count = len(image_path_list)
                spectrum_count = len(spectra_path_list)
                self.metadata_processor.clean_fits_header_tables(h5_connector)
                self.metadata_processor.create_fits_header_datasets(h5_connector, max_images=image_count,
                                                                    max_spectra=spectrum_count)
                h5_connector.set_image_count(image_count)
                h5_connector.set_spectrum_count(spectrum_count)
        self.mpi_helper.barrier()
        with self.parallel_connector as h5_connector:
            if self.rank == 0:
                self.distribute_work(h5_connector, image_path_list, self.metadata_processor)
                self.mpi_helper.barrier()
                self.distribute_work(h5_connector, spectra_path_list, self.metadata_processor)
            else:
                image_header_ds = get_image_header_dataset(h5_connector)
                image_header_ds_dtype = image_header_ds.dtype
                spectrum_header_ds = get_spectrum_header_dataset(h5_connector)
                spectrum_header_ds_dtype = spectrum_header_ds.dtype
                self.process_metadata_cache(h5_connector, image_header_ds, image_header_ds_dtype, "image_count")
                self.mpi_helper.barrier()
                self.process_metadata_cache(h5_connector, spectrum_header_ds, spectrum_header_ds_dtype,
                                            "spectrum_count")
            self.mpi_helper.barrier()

    def process_metadata_cache(self, h5_connector, header_ds, header_ds_dtype, count_type):
        status = self.mpi_helper.MPI.Status()
        path_list, offset = self.mpi_helper.receive_work_parsed(status)
        while status.Get_tag() != self.mpi_helper.KILL_TAG:
            try:
                inserted_cnt = self.metadata_processor.process_fits_headers(h5_connector, header_ds, header_ds_dtype,
                                                                            self.fits_image_path, path_list, offset)
            except Exception as e:
                self.logger.warning("Could not process %s, message: %s" % (path_list, str(e)))
                inserted_cnt = 0
            self.mpi_helper.comm.send(obj=inserted_cnt, tag=self.mpi_helper.FINISHED_TAG, dest=0)
            path_list, offset = self.mpi_helper.receive_work_parsed(status)


class ParallelDataBuilder(ParallelBuilder):
    def __init__(self, config: Config, h5_connector: H5Connector, mpi_helper: MPIHelper,
                 metadata_processor: MetadataProcessor, image_processor: ImageProcessor,
                 spectrum_processor: SpectrumProcessor, photometry: Photometry):
        super().__init__(config, h5_connector, mpi_helper)
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
                self.process_image_data(h5_connector, self.process_image)
            self.mpi_helper.barrier()
            if self.mpi_helper.rank == 0:
                spectra_paths = get_str_paths(get_spectrum_header_dataset(h5_connector))
                self.distribute_work(h5_connector, spectra_paths, self.spectrum_processor)
            else:
                self.process_spectra_data(h5_connector, self.process_spectrum)
            self.mpi_helper.barrier()

    @abstractmethod
    def process_image_data(self, h5_connector, process_func, *args, **kwargs):
        pass

    @abstractmethod
    def process_image(self, h5_connector, i, image_path, offset):
        pass

    @abstractmethod
    def process_spectra_data(self, h5_connector, process_func, *args, **kwargs):
        pass

    @abstractmethod
    def process_spectrum(self, h5_connector, i, spec_path, offset):
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

    def process_image_data(self, h5_connector, process_func, *args, **kwargs):
        self.process_path(h5_connector, process_func, *args, **kwargs)

    def process_image(self, h5_connector, i, image_path, offset):
        self.logger.debug("Rank %02d: Processing %s." % (self.mpi_helper.rank, image_path))
        metadata, data = self.photometry.get_multiple_resolution_image(image_path, self.config.IMG_ZOOM_CNT)
        self.write_data(metadata, data, h5_connector, image_path, self.image_processor, offset + i)

    def process_spectra_data(self, h5_connector, process_func, *args, **kwargs):
        self.process_path(h5_connector, process_func, *args, **kwargs)

    def process_spectrum(self, h5_connector, i, spec_path, offset):
        self.logger.debug("Rank %02d: Processing %s." % (self.mpi_helper.rank, spec_path))
        metadata, data = self.photometry.get_multiple_resolution_spectrum(
            spec_path, self.config.SPEC_ZOOM_CNT,
            apply_rebin=self.config.APPLY_REBIN,
            rebin_min=self.config.REBIN_MIN,
            rebin_max=self.config.REBIN_MAX,
            rebin_samples=self.config.REBIN_SAMPLES,
            apply_transmission=self.config.APPLY_TRANSMISSION_CURVE)
        self.write_data(metadata, data, h5_connector, spec_path, self.spectrum_processor, offset + i)


class ParallelSWMRDataBuilder(ParallelDataBuilder):
    def process_spectrum(self, h5_connector, i, image_path, offset):
        pass

    def process_image(self, h5_connector, i, image_path, offset):
        pass

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
            metadata = response["metadata"]
            data = response["data"]
            file_name = response["dataset_name"]
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
            {"metadata": dict(metadata), "data": data,
             "dataset_name": str(file_name)})
