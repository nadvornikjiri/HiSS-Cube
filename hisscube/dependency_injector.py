import string
from pathlib import Path

from hisscube.builders import SingleImageBuilder, SingleSpectrumBuilder, MetadataCacheBuilder, MetadataBuilder, \
    CBoosterMetadataBuilder, DataBuilder, LinkBuilder, MLCubeBuilder, VisualizationCubeBuilder
from hisscube.builders_parallel import ParallelMWMRDataBuilder, ParallelSWMRDataBuilder, ParallelMetadataCacheBuilder
from hisscube.processors.cube_ml import MLProcessor
from hisscube.processors.cube_visualization import VisualizationProcessor
from hisscube.processors.metadata import MetadataProcessor
from hisscube.processors.image import ImageProcessor
from hisscube.processors.metadata_strategy_cube_ml import TreeMLProcessorStrategy, DatasetMLProcessorStrategy
from hisscube.processors.metadata_strategy_cube_visualization import TreeVisualizationProcessorStrategy, \
    DatasetVisualizationProcessorStrategy
from hisscube.processors.spectrum import SpectrumProcessor
from hisscube.processors.metadata_strategy_dataset import DatasetStrategy
from hisscube.processors.metadata_strategy_tree import TreeStrategy
from hisscube.processors.metadata_strategy_image import TreeImageStrategy, DatasetImageStrategy
from hisscube.processors.metadata_strategy_spectrum import TreeSpectrumStrategy, DatasetSpectrumStrategy
from hisscube.utils.config import Config
from hisscube.utils.io import SerialH5Writer, ParallelH5Writer, CBoostedMetadataBuildWriter, SerialH5Reader
from hisscube.utils.io_strategy import SerialTreeIOStrategy, CBoostedTreeIOStrategy, SerialDatasetIOStrategy
from hisscube.utils.mpi_helper import MPIHelper
from hisscube.utils.photometry import Photometry


class HiSSCubeProvider:
    def __init__(self, h5_output_path, input_path=None, image_path=None, spectra_path=None, image_pattern=None,
                 spectra_pattern=None, config=None, image_list=None, spectra_list=None):
        if not image_path and input_path:
            image_path = Path(input_path).joinpath("images")
        if not spectra_path and input_path:
            spectra_path = Path(input_path).joinpath("spectra")
        if not config:
            self.config = Config()
        else:
            self.config = config
        self.image_list = image_list
        self.spectra_list = spectra_list
        self.photometry = Photometry()
        if self.config.METADATA_STRATEGY == "TREE":
            self.io_strategy = SerialTreeIOStrategy()
        else:
            self.io_strategy = SerialDatasetIOStrategy()

        self.c_boosted_strategy = CBoostedTreeIOStrategy()
        self.h5_serial_writer = SerialH5Writer(h5_output_path, self.config, self.io_strategy)
        self.h5_serial_reader = SerialH5Reader(h5_output_path, self.config, self.io_strategy)
        self.h5_c_boosted_serial_writer = CBoostedMetadataBuildWriter(h5_output_path, self.config,
                                                                      self.c_boosted_strategy)
        self.h5_parallel_writer = ParallelH5Writer(h5_output_path, self.config, self.io_strategy)

        self.mpi_helper = MPIHelper(self.config)
        self.processors: ProcessorProvider = ProcessorProvider(self.photometry, self.config, self.image_list,
                                                               self.spectra_list)
        self.serial_builders: SerialBuilderProvider = SerialBuilderProvider(image_path, spectra_path, self.config,
                                                                            self.h5_serial_writer,
                                                                            self.h5_c_boosted_serial_writer,
                                                                            self.processors, self.photometry,
                                                                            fits_image_pattern=image_pattern,
                                                                            fits_spectra_pattern=spectra_pattern)
        self.parallel_builders: ParallelBuilderProvider = ParallelBuilderProvider(image_path, spectra_path, self.config,
                                                                                  self.h5_serial_writer,
                                                                                  self.h5_parallel_writer,
                                                                                  self.processors,
                                                                                  self.mpi_helper, self.photometry,
                                                                                  fits_image_pattern=image_pattern,
                                                                                  fits_spectra_pattern=spectra_pattern)


class ProcessorProvider:
    def __init__(self, photometry, config, image_list=None, spectra_list=None):
        if config.METADATA_STRATEGY == "TREE":
            self.metadata_strategy = TreeStrategy(config)
            self.image_metadata_strategy = TreeImageStrategy(self.metadata_strategy, config, photometry)
            self.spectrum_metadata_strategy = TreeSpectrumStrategy(self.metadata_strategy, config, photometry)
            self.visualization_cube_strategy = TreeVisualizationProcessorStrategy(config, self.metadata_strategy)
            self.ml_cube_strategy = TreeMLProcessorStrategy(config, self.metadata_strategy, photometry)
        elif config.METADATA_STRATEGY == "DATASET":
            self.metadata_strategy = DatasetStrategy(config)
            self.image_metadata_strategy = DatasetImageStrategy(self.metadata_strategy, config, photometry)
            self.spectrum_metadata_strategy = DatasetSpectrumStrategy(self.metadata_strategy, config, photometry)
            self.visualization_cube_strategy = DatasetVisualizationProcessorStrategy(config, photometry,
                                                                                     self.metadata_strategy)
            self.ml_cube_strategy = DatasetMLProcessorStrategy(config, self.metadata_strategy, photometry)
        else:
            raise AttributeError(
                "Unsupported METADATA_STRATEGY %s, supported options are: TREE, DATASET." % config.METADATA_STRATEGY)
        self.metadata_processor = MetadataProcessor(config, photometry, self.metadata_strategy, image_list=image_list,
                                                    spectra_list=spectra_list)
        self.image_metadata_processor = ImageProcessor(config, self.metadata_processor,
                                                       self.image_metadata_strategy)
        self.spectrum_metadata_processor = SpectrumProcessor(config, self.metadata_processor,
                                                             self.spectrum_metadata_strategy)
        self.ml_cube_processor = MLProcessor(self.ml_cube_strategy)
        self.visualization_cube_processor = VisualizationProcessor(self.visualization_cube_strategy)


class SerialBuilderProvider:
    def __init__(self, fits_image_path: string, fits_spectra_path: string, config: Config,
                 h5_connector: SerialH5Writer, c_boosted_connector: CBoostedMetadataBuildWriter,
                 processors: ProcessorProvider,
                 photometry: Photometry, fits_image_pattern=None, fits_spectra_pattern=None):
        self.single_image_builder = SingleImageBuilder(config, h5_connector, processors.image_metadata_processor,
                                                       photometry)
        self.single_spectrum_builder = SingleSpectrumBuilder(config, h5_connector,
                                                             processors.spectrum_metadata_processor, photometry)
        self.metadata_cache_builder = MetadataCacheBuilder(fits_image_path, fits_spectra_path, config,
                                                           h5_connector, processors.metadata_processor,
                                                           fits_image_pattern=fits_image_pattern,
                                                           fits_spectra_pattern=fits_spectra_pattern)
        self.metadata_builder = MetadataBuilder(config, h5_connector, processors.image_metadata_processor,
                                                processors.spectrum_metadata_processor)
        self.c_boosted_metadata_builder = CBoosterMetadataBuilder(config, c_boosted_connector,
                                                                  processors.image_metadata_processor,
                                                                  processors.spectrum_metadata_processor)
        self.data_builder = DataBuilder(config, h5_connector, processors.image_metadata_processor,
                                        processors.spectrum_metadata_processor,
                                        self.single_image_builder, self.single_spectrum_builder)
        self.link_builder = LinkBuilder(config, h5_connector, processors.spectrum_metadata_processor)
        self.ml_cube_builder = MLCubeBuilder(config, h5_connector, processors.ml_cube_processor)
        self.visualization_cube_builder = VisualizationCubeBuilder(config, h5_connector,
                                                                   processors.visualization_cube_processor)


class ParallelBuilderProvider:
    def __init__(self, fits_image_path: string, fits_spectra_path: string, config: Config,
                 serial_connector: SerialH5Writer,
                 h5_connector: ParallelH5Writer, processors: ProcessorProvider,
                 mpi_helper: MPIHelper,
                 photometry: Photometry, fits_image_pattern=None, fits_spectra_pattern=None):
        self.metadata_cache_builder = ParallelMetadataCacheBuilder(fits_image_path, fits_spectra_path, config,
                                                                   serial_connector,
                                                                   h5_connector, mpi_helper,
                                                                   processors.metadata_processor,
                                                                   fits_image_pattern=fits_image_pattern,
                                                                   fits_spectra_pattern=fits_spectra_pattern)
        self.data_builder_MWMR = ParallelMWMRDataBuilder(config, h5_connector, mpi_helper,
                                                         processors.metadata_processor,
                                                         processors.image_metadata_processor,
                                                         processors.spectrum_metadata_processor,
                                                         photometry)
        self.data_builder_SWMR = ParallelSWMRDataBuilder(config, h5_connector, mpi_helper,
                                                         processors.metadata_processor,
                                                         processors.image_metadata_processor,
                                                         processors.spectrum_metadata_processor,
                                                         photometry)
