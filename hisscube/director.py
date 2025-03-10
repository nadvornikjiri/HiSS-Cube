from hisscube.utils.config import Config
from hisscube.utils.io import truncate


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
            truncate(self.h5_path, self.config)
            self.append_metadata_cache_builder()
            self.append_metadata_builder()
            self.append_data_builder(True, True, True)
            if self.config.CREATE_REFERENCES:
                self.append_link_builder()
            if self.config.CREATE_VISUALIZATION_CUBE:
                self.builders.append(self.serial_builders.visualization_cube_builder)
            if self.config.CREATE_ML_CUBE:
                self.append_ml_cube_builder()

        elif self.args.command == "update":
            if self.args.truncate:
                truncate(self.h5_path, self.config)
            if self.args.fits_metadata_cache:
                self.append_metadata_cache_builder()
            if self.args.metadata:
                self.append_metadata_builder()
            if self.args.data or self.args.data_image or self.args.data_spectrum:
                self.append_data_builder(self.args.data, self.args.data_image, self.args.data_spectrum)
            if self.args.link:
                self.append_link_builder()
            if self.args.visualization_cube:
                self.builders.append(self.serial_builders.visualization_cube_builder)
            if self.args.ml_cube:
                self.append_ml_cube_builder()
            if  self.args.sfr:
                self.builders.append(self.serial_builders.sfr_builder)

        for builder in self.builders:
            builder.build()

    def append_metadata_builder(self):
        if self.config.C_BOOSTER and self.config.METADATA_STRATEGY == "TREE" and not self.config.MPIO:
            self.builders.append(self.serial_builders.c_boosted_metadata_builder)
        elif self.config.MPIO:
            self.builders.append(self.parallel_builders.metadata_builder)
        else:
            self.builders.append(self.serial_builders.metadata_builder)

    def append_metadata_cache_builder(self):
        if self.config.MPIO:
            self.builders.append(self.parallel_builders.metadata_cache_builder)
        else:
            self.builders.append(self.serial_builders.metadata_cache_builder)

    def append_data_builder(self, all_data, image_data, spectrum_data):
        if self.config.MPIO:
            builder = None
            if self.config.PARALLEL_MODE == "MWMR":
                builder = self.parallel_builders.data_builder_MWMR
            elif self.config.PARALLEL_MODE == "SMWR":
                builder = self.parallel_builders.data_builder_SWMR
            if all_data:
                image_data = True
                spectrum_data = True
            builder.should_process_images = image_data
            builder.should_process_spectra = spectrum_data
            self.builders.append(builder)
        else:
            self.builders.append(self.serial_builders.data_builder)

    def append_link_builder(self):
        if self.config.MPIO:
            self.builders.append(self.parallel_builders.link_builder)
        else:
            self.builders.append(self.serial_builders.link_builder)

    def append_ml_cube_builder(self):
        if self.config.MPIO:
            self.builders.append(self.parallel_builders.ml_cube_builder)
        else:
            self.builders.append(self.serial_builders.ml_cube_builder)


