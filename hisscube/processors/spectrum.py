from hisscube.processors.metadata import get_str_path_list
from hisscube.processors.metadata_strategy_spectrum import SpectrumMetadataStrategy
from hisscube.utils.io import get_path_patterns, get_spectrum_header_dataset
from hisscube.utils.logging import HiSSCubeLogger


class SpectrumProcessor:
    def __init__(self, config, metadata_handler, metadata_strategy: SpectrumMetadataStrategy):
        self.metadata_strategy = metadata_strategy
        self.metadata_processor = metadata_handler
        self.metadata = None
        self.h5_connector = None
        self.config = config
        self.logger = HiSSCubeLogger.logger
        self.spec_cnt = 0
        self.spectrum_length = 0

    def update_fits_metadata_cache(self, h5_connector, spec_path, spec_pattern=None):
        self._set_connector(h5_connector)
        spec_pattern, spectra_pattern = get_path_patterns(self.config, None, spec_pattern)
        try:
            self.spec_cnt = self.h5_connector.get_spectrum_count()  # header datasets not created yet
        except KeyError:
            self.spec_cnt = 0
            self.metadata_processor.create_fits_header_datasets(h5_connector, max_spectra=1)
        spec_header_ds = get_spectrum_header_dataset(h5_connector)
        spec_path_list = get_str_path_list(spec_path, spectra_pattern, self.config.LIMIT_SPECTRA_COUNT,
                                           self.config.PATH_WAIT_TOTAL)
        self.spec_cnt += self.metadata_processor.write_fits_headers(h5_connector, spec_header_ds, spec_header_ds.dtype,
                                                                    spec_path,
                                                                    spec_path_list, offset=self.spec_cnt)
        self.h5_connector.file.attrs["spectrum_count"] = self.spec_cnt

    def get_resolution_groups(self, metadata, h5_connector):
        return self.metadata_strategy.get_resolution_groups(metadata, h5_connector)

    def write_metadata(self, h5_connector, no_attrs=False, no_datasets=False, range_min=None, range_max=None,
                       batch_size=None):
        self.metadata_strategy.write_metadata_multiple(h5_connector, no_attrs, no_datasets, range_min,
                                                       range_max, batch_size)

    def write_spectrum_metadata(self, h5_connector, fits_path, fits_header, no_attrs=False, no_datasets=False):
        self.metadata_strategy.write_metadata(h5_connector, fits_path, fits_header, no_attrs, no_datasets)

    def write_datasets(self, res_grp_list, data, file_name, offset=0, batch_i=0, batch_size=1, coordinates=None):
        return self.metadata_strategy.write_datasets(res_grp_list, data, file_name, offset, batch_i, batch_size,
                                                     coordinates)

    def link_spectra_to_images(self, h5_connector):
        self.metadata_strategy.link_spectra_to_images(h5_connector)

    def _set_connector(self, h5_connector):
        self.h5_connector = h5_connector
        self.metadata_processor.h5_connector = h5_connector
