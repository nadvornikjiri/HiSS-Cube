from hisscube.processors.metadata import get_str_path_list
from hisscube.processors.metadata_strategy_image import ImageMetadataStrategy
from hisscube.utils.io import get_path_patterns, get_image_header_dataset
from hisscube.utils.logging import HiSSCubeLogger


class ImageProcessor:
    def __init__(self, config, metadata_handler, metadata_strategy: ImageMetadataStrategy, logger: HiSSCubeLogger):
        self.h5_connector = None
        self.metadata_processor = metadata_handler
        self.logger = logger
        self.config = config
        self.metadata_strategy = metadata_strategy
        self.img_cnt = 0

    def update_fits_metadata_cache(self, h5_connector, image_path, image_pattern=None):
        self._set_connector(h5_connector)
        image_pattern, spectra_pattern = get_path_patterns(self.config, image_pattern, None)
        try:
            self.img_cnt = self.h5_connector.file.attrs["image_count"]  # header datasets not created yet
        except KeyError:
            self.img_cnt = 0
            self.metadata_processor.create_fits_header_datasets(h5_connector, max_images=1)
        image_header_ds = get_image_header_dataset(h5_connector)
        image_path_list = get_str_path_list(image_path, image_pattern, self.config.LIMIT_IMAGE_COUNT,
                                            self.config.PATH_WAIT_TOTAL)
        self.img_cnt += self.metadata_processor.write_fits_headers(h5_connector, image_header_ds, image_header_ds.dtype,
                                                                   image_path,
                                                                   image_path_list, offset=self.img_cnt)
        self.h5_connector.file.attrs["image_count"] = self.img_cnt

    def get_resolution_groups(self, metadata, h5_connector):
        yield from self.metadata_strategy.get_resolution_groups(metadata, h5_connector)

    def write_image_metadata(self, h5_connector, fits_path, fits_header, no_attrs=False, no_datasets=False):
        self.metadata_strategy.write_metadata(h5_connector, fits_path, fits_header, no_attrs, no_datasets)

    def write_metadata(self, h5_connector, no_attrs=False, no_datasets=False, range_min=None, range_max=None,
                       batch_size=None):
        self.metadata_strategy.write_metadata_multiple(h5_connector, no_attrs, no_datasets, range_min,
                                                       range_max, batch_size)

    def write_datasets(self, res_grp_list, data, file_name, offset=0, batch_i=0, batch_size=1):
        return self.metadata_strategy.write_datasets(res_grp_list, data, file_name, offset, batch_i, batch_size)

    def _set_connector(self, h5_connector):
        self.h5_connector = h5_connector
        self.metadata_processor.h5_connector = h5_connector
