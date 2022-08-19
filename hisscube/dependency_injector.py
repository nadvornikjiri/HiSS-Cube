from hisscube.builders import *
from hisscube.processors.data import DataProcessor, ImageDataProcessor, SpectrumDataProcessor
from hisscube.processors.metadata import MetadataProcessor
from hisscube.processors.metadata_image import ImageMetadataProcessor
from hisscube.processors.metadata_spectrum import SpectrumMetadataProcessor
from hisscube.utils.config import Config
from hisscube.utils.io import SerialH5Connector, ParallelH5Connector
from hisscube.utils.photometry import Photometry


class HiSSCubeProvider:
    def __init__(self, input_path, h5_output_path):
        fits_image_path = Path(input_path).joinpath("images")
        fits_spectra_path = Path(input_path).joinpath("spectra")
        self.config = Config()
        self.photometry = Photometry()
        self.h5_serial_writer = SerialH5Connector(h5_output_path)
        self.h5_c_boosted_serial_writer = CBoostedMetadataBuildConnector(h5_output_path)
        self.h5_parallel_writer = ParallelH5Connector(h5_output_path)

        self.mpi_helper = MPIHelper(self.config)
        self.processors = ProcessorProvider(self.photometry, self.config)
        self.serial_builders = SerialBuilderProvider(fits_image_path, fits_spectra_path, self.config,
                                                     self.h5_serial_writer, self.h5_c_boosted_serial_writer,
                                                     self.processors, self.photometry)
        self.parallel_builders = ParallelBuilderProvider(self.config, self.h5_parallel_writer, self.processors,
                                                         self.mpi_helper, self.photometry)


class ProcessorProvider:
    def __init__(self, photometry, config):
        self.metadata_processor = MetadataProcessor(config, photometry)
        self.image_metadata_processor = ImageMetadataProcessor(config, self.metadata_processor)
        self.spectrum_metadata_processor = SpectrumMetadataProcessor(config, self.metadata_processor)
        self.data_processor = DataProcessor(config)
        self.image_data_processor = ImageDataProcessor(config, self.data_processor)
        self.spectrum_data_processor = SpectrumDataProcessor(config, self.data_processor)
        self.ml_cube_processor = MLProcessor(config)
        self.visualization_cube_processor = VisualizationProcessor(config)


class SerialBuilderProvider:
    def __init__(self, fits_image_path: string, fits_spectra_path: string, config: Config,
                 h5_connector: SerialH5Connector, c_boosted_connector: CBoostedMetadataBuildConnector,
                 processors: ProcessorProvider,
                 photometry: Photometry):
        self.single_image_builder = SingleImageBuilder(config, h5_connector, processors.image_metadata_processor,
                                                       processors.image_data_processor, photometry)
        self.single_spectrum_builder = SingleSpectrumBuilder(config, h5_connector,
                                                             processors.spectrum_metadata_processor,
                                                             processors.spectrum_data_processor, photometry)
        self.metadata_cache_builder = MetadataCacheBuilder(fits_image_path, fits_spectra_path, config,
                                                           h5_connector, processors.metadata_processor)
        self.metadata_builder = MetadataBuilder(config, h5_connector, processors.image_metadata_processor,
                                                processors.spectrum_metadata_processor)
        self.c_boosted_metadata_builder = CBoosterMetadataBuilder(config, c_boosted_connector,
                                                                  processors.image_metadata_processor,
                                                                  processors.spectrum_metadata_processor)
        self.data_builder = DataBuilder(config, h5_connector, processors.image_metadata_processor,
                                        processors.spectrum_metadata_processor,
                                        processors.image_data_processor, processors.spectrum_data_processor,
                                        self.single_image_builder, self.single_spectrum_builder)
        self.link_builder = LinkBuilder(config, h5_connector, processors.spectrum_metadata_processor)
        self.ml_cube_builder = MLCubeBuilder(config, h5_connector, processors.ml_cube_processor)
        self.visualization_cube_builder = VisualizationCubeBuilder(config, h5_connector,
                                                                   processors.visualization_cube_processor)


class ParallelBuilderProvider:
    def __init__(self, config: Config,
                 h5_connector: ParallelH5Connector, processors: ProcessorProvider,
                 mpi_helper: MPIHelper,
                 photometry: Photometry):
        self.data_builder_MWMR = ParallelMWMRDataBuilder(config, h5_connector, mpi_helper,
                                                         processors.metadata_processor,
                                                         processors.image_metadata_processor,
                                                         processors.image_data_processor,
                                                         processors.spectrum_metadata_processor,
                                                         processors.spectrum_data_processor,
                                                         photometry)
        self.data_builder_SWMR = ParallelSWMRDataBuilder(config, h5_connector, mpi_helper,
                                                         processors.metadata_processor,
                                                         processors.image_metadata_processor,
                                                         processors.image_data_processor,
                                                         processors.spectrum_metadata_processor,
                                                         processors.spectrum_data_processor,
                                                         photometry)
