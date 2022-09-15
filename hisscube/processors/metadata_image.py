import ujson

from hisscube.processors.metadata_strategy_image import ImageMetadataStrategy
from hisscube.utils.io import get_path_patterns, get_image_header_dataset
from hisscube.utils.logging import HiSSCubeLogger, log_timing


class ImageMetadataProcessor:
    def __init__(self, config, metadata_handler, metadata_strategy: ImageMetadataStrategy):
        self.h5_connector = None
        self.metadata_processor = metadata_handler
        self.config = config
        self.logger = HiSSCubeLogger.logger
        self.metadata_strategy = metadata_strategy
        self.img_cnt = 0

    def get_resolution_groups(self, metadata, h5_connector):
        return self.metadata_strategy.get_resolution_groups(metadata, h5_connector)

    def update_fits_metadata_cache(self, h5_connector, image_path, image_pattern=None):
        self._set_connector(h5_connector)
        image_pattern, spectra_pattern = get_path_patterns(self.config, image_pattern, None)
        try:
            self.img_cnt = self.h5_connector.file.attrs["image_count"]  # header datasets not created yet
        except KeyError:
            self.img_cnt = 0
            self.metadata_processor.create_fits_header_datasets()
        image_header_ds = get_image_header_dataset(h5_connector)
        self.img_cnt += self.metadata_processor.write_fits_headers(image_header_ds, image_header_ds.dtype, image_path,
                                                                   image_pattern,
                                                                   self.config.LIMIT_IMAGE_COUNT, offset=self.img_cnt)
        self.h5_connector.file.attrs["image_count"] = self.img_cnt

    def write_image_metadata(self, h5_connector, fits_path, fits_header, no_attrs=False, no_datasets=False):
        self._set_connector(h5_connector)
        metadata = ujson.loads(fits_header)
        self._write_parsed_image_metadata(metadata, fits_path, no_attrs, no_datasets)

    def _set_connector(self, h5_connector):
        self.h5_connector = h5_connector
        self.metadata_processor.h5_connector = h5_connector
        self.metadata_strategy.h5_connector = h5_connector

    def write_images_metadata(self, h5_connector, no_attrs=False, no_datasets=False):
        self._set_connector(h5_connector)
        fits_headers = self.h5_connector.file["/fits_images_metadata"]
        self._write_image_metadata_from_cache(h5_connector, fits_headers, no_attrs, no_datasets)

    def _write_image_metadata_from_cache(self, h5_connector, fits_headers, no_attrs, no_datasets):
        self.img_cnt = 0
        self.h5_connector.fits_total_cnt = 0
        for fits_path, header in fits_headers:
            if not fits_path:  # end of data
                break
            self._write_metadata_from_header(h5_connector, fits_path, header, no_attrs, no_datasets)
            if self.img_cnt >= self.config.LIMIT_IMAGE_COUNT:
                break
        self.h5_connector.set_attr(self.h5_connector.file, "image_count", self.img_cnt)

    @log_timing("process_image_metadata")
    def _write_metadata_from_header(self, h5_connector, fits_path, header, no_attrs, no_datasets):
        self._set_connector(h5_connector)
        fits_path = fits_path.decode('utf-8')
        if self.img_cnt % 100 == 0 and self.img_cnt / 100 > 0:
            self.logger.info("Image cnt: %05d" % self.img_cnt)
        try:
            self.write_image_metadata(h5_connector, fits_path, header, no_attrs=no_attrs, no_datasets=no_datasets)
            self.img_cnt += 1
            self.h5_connector.fits_total_cnt += 1
        except RuntimeError as e:
            self.logger.warning(
                "Unable to ingest image %s, message: %s" % (fits_path, str(e)))
            raise e

    def _write_parsed_image_metadata(self, metadata, fits_path, no_attrs, no_datasets):
        self.metadata_strategy.write_parsed_image_metadata(metadata, fits_path, no_attrs, no_datasets)




