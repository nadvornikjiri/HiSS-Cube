import math
import string
import sys
import traceback
from abc import ABCMeta, abstractmethod
from pathlib import Path

import h5py
from tqdm.auto import tqdm

from hisscube.builders import Builder
from hisscube.processors.cube_ml import MLProcessor
from hisscube.processors.image import ImageProcessor
from hisscube.processors.metadata import MetadataProcessor
from hisscube.processors.metadata_strategy import require_zoom_grps
from hisscube.processors.metadata_strategy_cube_ml import DatasetMLProcessorStrategy
from hisscube.processors.metadata_strategy_dataset import get_index_datasets, get_wcs_datasets
from hisscube.processors.metadata_strategy_image import DatasetImageStrategy
from hisscube.processors.metadata_strategy_spectrum import DatasetSpectrumStrategy
from hisscube.processors.spectrum import SpectrumProcessor
from hisscube.utils.config import Config
from hisscube.utils.io import H5Connector, get_str_paths, get_image_header_dataset, get_spectrum_header_dataset, \
    SerialH5Writer, ParallelH5Writer
from hisscube.utils.logging import wrap_tqdm
from hisscube.utils.mpi_helper import MPIHelper, chunks
from hisscube.utils.photometry import Photometry


class ParallelBuilder(Builder, metaclass=ABCMeta):

    def __init__(self, config: Config, h5_connector: H5Connector, mpi_helper: MPIHelper):
        super().__init__(config, h5_connector)
        self.mpi_helper = mpi_helper
        self.comm_buffer = None

    def distribute_work(self, h5_connector, path_list, processor, batch_size, desc, total=None):
        status = self.mpi_helper.MPI.Status()
        batches = chunks(path_list, batch_size)
        offset = 0
        with open("output.txt", "a") as output_file:
            pbar = tqdm(desc=desc, total=total, smoothing=0, file=output_file)
            for i, batch in enumerate(batches, 1):
                batch = list(batch)
                next_batch_size = len(batch)
                if i < self.mpi_helper.size:
                    self.mpi_helper.send_work(batch, dest=i, offset=offset)
                    pbar.update(next_batch_size)
                    self.mpi_helper.active_workers += 1
                else:
                    self.mpi_helper.wait_for_message(source=self.mpi_helper.MPI.ANY_SOURCE,
                                                     tag=self.mpi_helper.FINISHED_TAG, status=status)
                    self.process_response(h5_connector, processor, status)
                    self.mpi_helper.send_work(batch, status.Get_source(), offset=offset)
                    pbar.update(next_batch_size)
                offset += next_batch_size
            for i in range(1, self.mpi_helper.size):
                self.mpi_helper.send_work_finished(dest=i)
            for i in range(0, self.mpi_helper.active_workers):
                self.process_response(h5_connector, processor, status)
            self.mpi_helper.active_workers = 0
            pbar.close()
        return offset

    def process_response(self, h5_connector, processor, status):
        self.comm_buffer = self.mpi_helper.comm.recv(source=self.mpi_helper.MPI.ANY_SOURCE,
                                                     tag=self.mpi_helper.FINISHED_TAG,
                                                     status=status)
        self.h5_connector.fits_total_cnt += 1
        self.logger.debug(
            "Received response from. dest %02d: %d " % (status.Get_source(), self.mpi_helper.sent_work_cnt))

    def process_path(self, h5_connector, process_func, *args, **kwargs):
        status = self.mpi_helper.MPI.Status()
        path_list, offset = self.mpi_helper.receive_work_parsed(status)
        while status.Get_tag() != self.mpi_helper.KILL_TAG:
            batch_size = len(path_list)
            for i, path in enumerate(path_list):
                try:
                    process_func(h5_connector, i, path, offset, batch_size, *args, **kwargs)
                except Exception as e:
                    self.logger.warning("Could not process %s, message: %s" % (path, str(e)))
                    if self.config.LOG_LEVEL == "DEBUG":
                        self.logger.debug(traceback.format_exc())
                        raise e
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
        self.image_count = 0
        self.spectrum_count = 0

    def build(self):
        if self.rank == 0:
            with self.h5_connector as h5_connector:
                self.metadata_processor.open_csv_files()
                image_path_list, spectra_path_list = self.metadata_processor.parse_paths(self.fits_image_path,
                                                                                         self.fits_image_pattern,
                                                                                         self.fits_spectra_path,
                                                                                         self.fits_spectra_pattern)
                try:
                    image_count = len(image_path_list)
                    spectrum_count = len(spectra_path_list)
                except TypeError:
                    image_count = self.config.LIMIT_IMAGE_COUNT
                    spectrum_count = self.config.LIMIT_SPECTRA_COUNT
                self.metadata_processor.clean_fits_header_tables(h5_connector)
                self.metadata_processor.create_fits_header_datasets(h5_connector, max_images=image_count,
                                                                    max_spectra=spectrum_count)
                h5_connector.set_image_count(image_count)
                h5_connector.set_spectrum_count(spectrum_count)
        self.mpi_helper.barrier()
        with self.parallel_connector as h5_connector:
            if self.rank == 0:
                self.image_count = self.distribute_work(h5_connector, image_path_list, self.metadata_processor,
                                                        self.config.FITS_HEADER_BATCH_SIZE, "Image headers")
                self.mpi_helper.sent_work_cnt = 0
                self.mpi_helper.barrier()
                self.spectrum_count = self.distribute_work(h5_connector, spectra_path_list, self.metadata_processor,
                                                           self.config.FITS_HEADER_BATCH_SIZE, "Spectra headers")
            else:
                image_header_ds = get_image_header_dataset(h5_connector)
                image_header_ds_dtype = image_header_ds.dtype
                spectrum_header_ds = get_spectrum_header_dataset(h5_connector)
                spectrum_header_ds_dtype = spectrum_header_ds.dtype
                self.process_metadata_cache(h5_connector, image_header_ds, image_header_ds_dtype, "image")
                self.mpi_helper.barrier()
                self.process_metadata_cache(h5_connector, spectrum_header_ds, spectrum_header_ds_dtype, "spectrum")
        self.mpi_helper.barrier()
        if self.rank == 0:
            with self.h5_connector as h5_connector:
                h5_connector.set_image_count(self.image_count)
                h5_connector.set_spectrum_count(self.spectrum_count)
                image_header_ds = get_image_header_dataset(h5_connector)
                spectrum_header_ds = get_spectrum_header_dataset(h5_connector)
                image_header_ds.resize(self.image_count, axis=0)
                spectrum_header_ds.resize(self.spectrum_count, axis=0)
            self.metadata_processor.close_csv_files()

    def process_metadata_cache(self, h5_connector, header_ds, header_ds_dtype, data_type):
        status = self.mpi_helper.MPI.Status()
        path_list, offset = self.mpi_helper.receive_work_parsed(status)
        while status.Get_tag() != self.mpi_helper.KILL_TAG:
            try:
                inserted_cnt = self.metadata_processor.process_fits_headers(h5_connector, header_ds, header_ds_dtype,
                                                                            path_list, data_type, offset)
            except Exception as e:
                self.logger.warning("Could not process %s, message: %s" % (path_list, str(e)))
                inserted_cnt = 0
                if self.config.LOG_LEVEL == "DEBUG":
                    self.logger.debug(traceback.format_exc())
                    raise e
            self.mpi_helper.comm.send(obj=inserted_cnt, tag=self.mpi_helper.FINISHED_TAG, dest=0)
            path_list, offset = self.mpi_helper.receive_work_parsed(status)


class ParallelMetadataBuilder(ParallelBuilder):
    def __init__(self, config: Config, serial_h5_connector: SerialH5Writer, parallel_h5_connector: ParallelH5Writer,
                 mpi_helper: MPIHelper,
                 metadata_processor: MetadataProcessor, image_metadata_strategy: DatasetImageStrategy,
                 spectrum_metadata_strategy: DatasetSpectrumStrategy, image_processor: ImageProcessor,
                 spectrum_processor: SpectrumProcessor):
        super().__init__(config, serial_h5_connector, mpi_helper)
        self.metadata_processor = metadata_processor
        self.image_strategy = image_metadata_strategy
        self.spectrum_strategy = spectrum_metadata_strategy
        self.image_processor = image_processor
        self.spectrum_processor = spectrum_processor
        self.parallel_connector = parallel_h5_connector
        self.image_count = 0
        self.spectrum_count = 0

    def build(self):
        if self.rank == 0:
            with self.h5_connector as h5_connector:
                self.image_strategy.recreate_datasets(h5_connector)
                self.spectrum_strategy.recreate_datasets(h5_connector)
                self.image_count = h5_connector.get_image_count()
                self.spectrum_count = h5_connector.get_spectrum_count()
        self.mpi_helper.barrier()
        with self.parallel_connector as h5_connector:
            if self.rank == 0:
                image_range = range(self.image_count)
                self.distribute_work(h5_connector, image_range, self.metadata_processor,
                                     self.config.METADATA_BATCH_SIZE, "Image metadata", total=self.image_count)
                self.mpi_helper.sent_work_cnt = 0
                self.mpi_helper.barrier()
                spectrum_range = range(self.spectrum_count)
                self.distribute_work(h5_connector, spectrum_range, self.metadata_processor,
                                     self.config.METADATA_BATCH_SIZE, "Spectra metadata", total=self.spectrum_count)
            else:
                self.process_metadata(h5_connector, "image")
                self.mpi_helper.barrier()
                self.image_strategy.clear_buffers()
                self.process_metadata(h5_connector, "spectrum")
                self.spectrum_strategy.clear_buffers()
        self.mpi_helper.barrier()
        if self.rank == 0:
            with self.h5_connector as h5_connector:
                self.image_strategy.sort_indices(h5_connector)
                self.spectrum_strategy.sort_indices(h5_connector)

    def process_metadata(self, h5_connector, data_type):
        status = self.mpi_helper.MPI.Status()
        range_list, offset = self.mpi_helper.receive_work_parsed(status)
        while status.Get_tag() != self.mpi_helper.KILL_TAG:
            batch_size = len(range_list)
            try:
                if data_type == "image":
                    inserted_cnt = self.image_processor.write_metadata(h5_connector, range_min=offset,
                                                                       range_max=offset + batch_size,
                                                                       batch_size=batch_size)
                if data_type == "spectrum":
                    inserted_cnt = self.spectrum_processor.write_metadata(h5_connector, range_min=offset,
                                                                          range_max=offset + batch_size,
                                                                          batch_size=batch_size)
            except Exception as e:
                self.logger.warning("Could not process %s, message: %s" % (range_list, str(e)))
                inserted_cnt = 0
                if self.config.LOG_LEVEL == "DEBUG":
                    self.logger.debug(traceback.format_exc())
                    raise e
            self.mpi_helper.comm.send(obj=inserted_cnt, tag=self.mpi_helper.FINISHED_TAG, dest=0)
            range_list, offset = self.mpi_helper.receive_work_parsed(status)


class ParallelDataBuilder(ParallelBuilder):
    def __init__(self, config: Config, h5_connector: H5Connector, mpi_helper: MPIHelper,
                 metadata_processor: MetadataProcessor, image_processor: ImageProcessor,
                 spectrum_processor: SpectrumProcessor, photometry: Photometry):
        super().__init__(config, h5_connector, mpi_helper)
        self.metadata_processor = metadata_processor
        self.image_processor = image_processor
        self.spectrum_processor = spectrum_processor
        self.photometry = photometry
        self.should_process_images = True
        self.should_process_spectra = True

    def build(self):
        self.mpi_helper.barrier()
        with self.h5_connector as h5_connector:
            if self.should_process_images:
                self.build_image_data(h5_connector)
            if self.should_process_spectra:
                self.build_spectrum_data(h5_connector)

    def build_spectrum_data(self, h5_connector):
        if self.mpi_helper.rank == 0:
            spec_cnt = h5_connector.get_spectrum_count()
            spectra_paths = get_str_paths(get_spectrum_header_dataset(h5_connector))
            self.distribute_work(h5_connector, spectra_paths, self.spectrum_processor,
                                 self.config.SPECTRUM_DATA_BATCH_SIZE, "Spectrum data", total=spec_cnt)
        else:
            self.process_spectra_data(h5_connector, self.process_spectrum)
            self.spectrum_processor.metadata_strategy.clear_buffers()
        self.mpi_helper.barrier()

    def build_image_data(self, h5_connector):
        if self.rank == 0:
            img_cnt = h5_connector.get_image_count()
            image_paths = get_str_paths(get_image_header_dataset(h5_connector))
            self.distribute_work(h5_connector, image_paths, self.image_processor, self.config.IMAGE_DATA_BATCH_SIZE,
                                 "Image data", total=img_cnt)
        else:
            self.process_image_data(h5_connector, self.process_image)
            self.image_processor.metadata_strategy.clear_buffers()
        self.mpi_helper.barrier()

    @abstractmethod
    def process_image_data(self, h5_connector, process_func, *args, **kwargs):
        pass

    @abstractmethod
    def process_image(self, h5_connector, i, image_path, offset, batch_size):
        pass

    @abstractmethod
    def process_spectra_data(self, h5_connector, process_func, *args, **kwargs):
        pass

    @abstractmethod
    def process_spectrum(self, h5_connector, i, spec_path, offset, batch_size):
        pass


class ParallelMWMRDataBuilder(ParallelDataBuilder):
    def __init__(self, config: Config, h5_connector: H5Connector, mpi_helper: MPIHelper,
                 metadata_processor: MetadataProcessor, image_processor: ImageProcessor,
                 spectrum_processor: SpectrumProcessor, photometry: Photometry):
        super().__init__(config, h5_connector, mpi_helper, metadata_processor, image_processor,
                         spectrum_processor, photometry)

    @staticmethod
    def write_data(metadata, data, h5_connector, fits_path, processor, offset, i, batch_size):
        res_grps = processor.get_resolution_groups(metadata, h5_connector)
        file_name = Path(fits_path).name
        processor.write_datasets(res_grps, data, file_name, offset=offset, batch_i=i, batch_size=batch_size)

    def process_image_data(self, h5_connector, process_func, *args, **kwargs):
        self.process_path(h5_connector, process_func, *args, **kwargs)

    def process_image(self, h5_connector, i, image_path, offset, batch_size):
        self.logger.debug("Rank %02d: Processing %s." % (self.mpi_helper.rank, image_path))
        metadata, data = self.photometry.get_multiple_resolution_image(image_path, self.config.IMG_ZOOM_CNT)
        self.write_data(metadata, data, h5_connector, image_path, self.image_processor, offset, i, batch_size)

    def process_spectra_data(self, h5_connector, process_func, *args, **kwargs):
        self.process_path(h5_connector, process_func, *args, **kwargs)

    def process_spectrum(self, h5_connector, i, spec_path, offset, batch_size):
        self.logger.debug("Rank %02d: Processing %s." % (self.mpi_helper.rank, spec_path))
        metadata, data = self.photometry.get_multiple_resolution_spectrum(
            spec_path, self.config.SPEC_ZOOM_CNT,
            apply_rebin=self.config.APPLY_REBIN,
            rebin_min=self.config.REBIN_MIN,
            rebin_max=self.config.REBIN_MAX,
            rebin_samples=self.config.REBIN_SAMPLES,
            apply_transmission=self.config.APPLY_TRANSMISSION_CURVE)
        self.write_data(metadata, data, h5_connector, spec_path, self.spectrum_processor, offset, i, batch_size)


class ParallelSWMRDataBuilder(ParallelDataBuilder):
    def process_spectrum(self, h5_connector, i, image_path, offset, batch_size):
        pass

    def process_image(self, h5_connector, i, image_path, offset, batch_size):
        pass

    def __init__(self, config: Config, h5_connector: H5Connector, mpi_helper: MPIHelper,
                 metadata_processor: MetadataProcessor, image_processor: ImageProcessor,
                 spectrum_processor: SpectrumProcessor, photometry: Photometry):
        super().__init__(config, h5_connector, mpi_helper, metadata_processor, image_processor,
                         spectrum_processor, photometry)
        self.comm_buffer = bytearray(
            self.config.IMAGE_DATA_BATCH_SIZE * 100 * 1024 * 1024)  # 100 MBs for one image

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


class ParallelLinkBuilder(ParallelBuilder):
    def __init__(self, config: Config, serial_h5_connector: SerialH5Writer, parallel_h5_connector: ParallelH5Writer,
                 mpi_helper: MPIHelper,
                 spectrum_metadata_strategy: DatasetSpectrumStrategy,
                 spectrum_processor: SpectrumProcessor):
        super().__init__(config, serial_h5_connector, mpi_helper)
        self.spectrum_strategy = spectrum_metadata_strategy
        self.spectrum_processor = spectrum_processor
        self.parallel_connector = parallel_h5_connector

    def build(self):
        if self.rank == 0:
            with self.h5_connector as serial_connector:
                spectrum_count = serial_connector.get_spectrum_count()
                spectrum_zoom_groups = require_zoom_grps("spectra", serial_connector, self.config.SPEC_ZOOM_CNT)
                self.spectrum_processor.recreate_datasets(serial_connector, spectrum_count, spectrum_zoom_groups)
        self.mpi_helper.barrier()
        with self.parallel_connector as h5_connector:
            if self.rank == 0:
                spectrum_range = range(spectrum_count)
                self.distribute_work(h5_connector, spectrum_range, self.spectrum_processor,
                                     self.config.LINK_BATCH_SIZE, "Linking spectra", total=spectrum_count)
            else:

                image_db_index_uncached = get_index_datasets(h5_connector, "images", self.config.IMG_ZOOM_CNT,
                                                    self.config.SPARSE_CUBE_NAME)[0]
                image_wcs_datasets_uncached = get_wcs_datasets(h5_connector, "images", self.config.IMG_ZOOM_CNT,
                                                      self.config.SPARSE_CUBE_NAME)
                if self.config.CACHE_INDEX_FOR_LINKING:
                    image_db_index = image_db_index_uncached[:]
                else:
                    image_db_index = image_db_index_uncached
                if self.config.CACHE_WCS_FOR_LINKING:
                    image_wcs_datasets = [ds[:] for ds in image_wcs_datasets_uncached]
                else:
                    image_wcs_datasets = image_wcs_datasets_uncached
                self.link(h5_connector, image_db_index, image_wcs_datasets)
                self.spectrum_strategy.clear_buffers()
            self.logger.error("Ref count for h5 file: %d " % h5py.h5i.get_ref(h5_connector.file.id))
        self.mpi_helper.barrier()

    def link(self, h5_connector, image_db_index, image_wcs_data):
        status = self.mpi_helper.MPI.Status()
        range_list, offset = self.mpi_helper.receive_work_parsed(status)
        while status.Get_tag() != self.mpi_helper.KILL_TAG:
            batch_size = len(range_list)
            try:
                inserted_cnt = self.spectrum_processor.link_spectra_to_images(h5_connector, offset,
                                                                              offset + batch_size,
                                                                              batch_size, image_db_index,
                                                                              image_wcs_data)
            except Exception as e:
                self.logger.warning("Could not process %s, message: %s" % (range_list, str(e)))
                if self.config.LOG_LEVEL == "DEBUG":
                    self.logger.debug(traceback.format_exc())
                    raise e
            self.mpi_helper.comm.send(obj=inserted_cnt, tag=self.mpi_helper.FINISHED_TAG, dest=0)
            range_list, offset = self.mpi_helper.receive_work_parsed(status)


class ParallelMLCubeBuilder(ParallelBuilder):
    def __init__(self, config: Config, serial_h5_connector: SerialH5Writer, parallel_h5_connector: ParallelH5Writer,
                 mpi_helper: MPIHelper,
                 metadata_strategy: DatasetMLProcessorStrategy,
                 ml_processor: MLProcessor):
        super().__init__(config, serial_h5_connector, mpi_helper)
        self.spectrum_strategy = metadata_strategy
        self.ml_processor = ml_processor
        self.parallel_connector = parallel_h5_connector
        self.inserted_target_counts = None
        self.is_building_finished = False
        self.total_target_count = 0

    def build(self):
        if self.rank == 0:
            with self.h5_connector as serial_connector:
                dense_grp, target_count, target_spatial_indices = self.ml_processor.get_targets(serial_connector)
                if target_count > 0:
                    final_zoom = self.ml_processor.recreate_datasets(serial_connector, dense_grp, target_count)
                    self.inserted_target_counts = [(0, 0)] * math.ceil(target_count / self.config.ML_BATCH_SIZE)
        self.mpi_helper.barrier()
        with self.parallel_connector as h5_connector:
            if self.rank == 0:
                self.distribute_work(h5_connector, target_spatial_indices, self.ml_processor,
                                     self.config.ML_BATCH_SIZE, "ML cube spectra", total=target_count)
            else:
                self.build_ml_cube(h5_connector)
                self.spectrum_strategy.clear_buffers()
        self.mpi_helper.barrier()
        if self.rank == 0:
            with self.h5_connector as serial_connector:
                dense_grp = serial_connector.get_dense_group()
                self.ml_processor.recreate_copy_datasets(serial_connector, dense_grp, target_count)
        self.mpi_helper.barrier()
        self.is_building_finished = True
        with self.parallel_connector as h5_connector:
            if self.rank == 0:
                new_offsets = self.prefix_sum(self.inserted_target_counts)
                self.distribute_work(h5_connector, new_offsets, self.ml_processor,
                                     1, "Shrinking ML cube", total=len(new_offsets))
                for old_offset, cnt, new_offset in new_offsets:
                    self.total_target_count += cnt
            else:
                self.copy_slice(h5_connector)
        self.mpi_helper.barrier()
        if self.rank == 0:
            with self.h5_connector as serial_connector:
                self.ml_processor.merge_datasets(serial_connector)
                self.ml_processor.shrink_datasets(final_zoom, serial_connector, self.total_target_count)
        self.mpi_helper.barrier()

    def build_ml_cube(self, h5_connector):
        status = self.mpi_helper.MPI.Status()
        target_spatial_indices, offset = self.mpi_helper.receive_work_parsed(status)
        while status.Get_tag() != self.mpi_helper.KILL_TAG:
            batch_size = len(target_spatial_indices)
            inserted_cnt = 0
            try:
                dense_grp, spectra_index_spec_ids_orig_zoom = self.ml_processor.get_entry_points(h5_connector)
                inserted_cnt = self.ml_processor.process_data(h5_connector, spectra_index_spec_ids_orig_zoom,
                                                              target_spatial_indices, offset,
                                                              offset + batch_size,
                                                              batch_size)
            except Exception as e:
                self.logger.warning("Could not process %s, message: %s" % (target_spatial_indices, str(e)))
                if self.config.LOG_LEVEL == "DEBUG":
                    self.logger.debug(traceback.format_exc())
                    raise e
            self.mpi_helper.comm.send(obj=(offset, inserted_cnt), tag=self.mpi_helper.FINISHED_TAG, dest=0)
            target_spatial_indices, offset = self.mpi_helper.receive_work_parsed(status)

    def process_response(self, h5_connector, processor, status):
        offset, inserted_cnt = self.mpi_helper.comm.recv(source=self.mpi_helper.MPI.ANY_SOURCE,
                                                         tag=self.mpi_helper.FINISHED_TAG,
                                                         status=status)
        job_idx = int(offset / self.config.ML_BATCH_SIZE)
        if not self.is_building_finished:
            self.inserted_target_counts[job_idx] = offset, inserted_cnt
        self.h5_connector.fits_total_cnt += 1
        self.logger.debug(
            "Received response from. dest %02d: %d " % (status.Get_source(), self.mpi_helper.sent_work_cnt))

    def copy_slice(self, h5_connector):
        status = self.mpi_helper.MPI.Status()
        copy_batch = self.mpi_helper.receive_work_parsed(status)
        while status.Get_tag() != self.mpi_helper.KILL_TAG:
            try:
                msg, offset = copy_batch
                for old_offset, cnt, new_offset in msg:
                    self.ml_processor.copy_slice(h5_connector, old_offset, cnt, new_offset)
            except Exception as e:
                self.logger.warning("Could not copy slice: %s, message: %s" % (
                    copy_batch, str(e)))
                print(traceback.format_exc())
            self.mpi_helper.comm.send(obj=(-1, -1), tag=self.mpi_helper.FINISHED_TAG, dest=0)
            copy_batch = self.mpi_helper.receive_work_parsed(status)

    @staticmethod
    def prefix_sum(inserted_target_counts):
        to_be_copied = []
        new_offset = 0
        for old_offset, cnt in inserted_target_counts:
            if cnt > 0:
                to_be_copied.append((old_offset, cnt, new_offset))
            new_offset += cnt
        return to_be_copied
