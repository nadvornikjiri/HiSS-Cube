import fitsio
import ujson

from hisscube.processors.metadata_strategy_spectrum import SpectrumMetadataStrategy
from hisscube.utils.io import get_path_patterns, get_spectrum_header_dataset
from hisscube.utils.logging import HiSSCubeLogger, log_timing


class SpectrumMetadataProcessor:
    def __init__(self, config, metadata_handler, metadata_strategy: SpectrumMetadataStrategy):
        self.metadata_strategy = metadata_strategy
        self.metadata_processor = metadata_handler
        self.metadata = None
        self.h5_connector = None
        self.config = config
        self.logger = HiSSCubeLogger.logger
        self.spec_cnt = 0
        self.spectrum_length = 0

    def get_resolution_groups(self, metadata, h5_connector):
        return self.metadata_strategy.get_resolution_groups(metadata, h5_connector)

    def update_fits_metadata_cache(self, h5_connector, spec_path, spec_pattern=None):
        self._set_connector(h5_connector)
        spec_pattern, spectra_pattern = get_path_patterns(self.config, None, spec_pattern)
        try:
            self.spec_cnt = self.h5_connector.file.attrs["spectra_count"]  # header datasets not created yet
        except KeyError:
            self.spec_cnt = 0
            self.metadata_processor.create_fits_header_datasets()
        spec_header_ds = get_spectrum_header_dataset(h5_connector)
        self.spec_cnt += self.metadata_processor.write_fits_headers(spec_header_ds, spec_header_ds.dtype, spec_path,
                                                                    spectra_pattern,
                                                                    self.config.LIMIT_SPECTRA_COUNT,
                                                                    offset=self.spec_cnt)
        self.h5_connector.file.attrs["spectra_count"] = self.spec_cnt

    def write_spectra_metadata(self, h5_connector, no_attrs=False, no_datasets=False):
        self._set_connector(h5_connector)
        fits_headers = self.h5_connector.file["/fits_spectra_metadata"]
        self._write_spectra_metadata_from_cache(h5_connector, fits_headers, no_attrs, no_datasets)

    def write_spectrum_metadata(self, h5_connector, fits_path, fits_header, no_attrs=False, no_datasets=False):
        self._set_connector(h5_connector)
        metadata = ujson.loads(fits_header)
        self._write_parsed_spectrum_metadata(metadata, fits_path, no_attrs, no_datasets)

    def link_spectra_to_images(self, h5_connector):
        self.metadata_strategy.link_spectra_to_images(h5_connector)

    def _set_connector(self, h5_connector):
        self.h5_connector = h5_connector
        self.metadata_processor.h5_connector = h5_connector
        self.metadata_strategy.h5_connector = h5_connector

    def _write_spectra_metadata_from_cache(self, h5_connector, fits_headers, no_attrs, no_datasets):
        self.spec_cnt = 0
        self.h5_connector.fits_total_cnt = 0
        for fits_path, header in fits_headers:
            if not fits_path:  # end of data
                break
            self._write_spectrum_metadata_from_header(h5_connector, fits_path, header, no_attrs, no_datasets)
            if self.spec_cnt >= self.config.LIMIT_SPECTRA_COUNT:
                break
        self.h5_connector.set_attr(self.h5_connector.file, "spectra_count", self.spec_cnt)

    @log_timing("process_spectrum_metadata")
    def _write_spectrum_metadata_from_header(self, h5_connector, fits_path, header, no_attrs, no_datasets):
        fits_path = fits_path.decode('utf-8')
        self._set_connector(h5_connector)
        if self.spec_cnt % 100 == 0 and self.spec_cnt / 100 > 0:
            self.logger.info("Spectra cnt: %05d" % self.spec_cnt)
        try:
            self.write_spectrum_metadata(h5_connector, fits_path, header, no_attrs, no_datasets)
            self.spec_cnt += 1
            self.h5_connector.fits_total_cnt += 1
        except ValueError as e:
            self.logger.warning(
                "Unable to ingest spectrum %s, message: %s" % (fits_path, str(e)))

    def _write_parsed_spectrum_metadata(self, metadata, fits_path, no_attrs, no_datasets):
        if self.config.APPLY_REBIN is False:
            spectrum_length = fitsio.read_header(fits_path, 1)["NAXIS2"]
        else:
            spectrum_length = self.config.REBIN_SAMPLES
        self.metadata_strategy.write_parsed_spectrum_metadata(metadata, spectrum_length, fits_path, no_attrs,
                                                              no_datasets)


