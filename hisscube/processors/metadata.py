import itertools
import pathlib

import fitsio
import h5py
import numpy as np
import ujson
from tqdm.auto import tqdm

from hisscube.processors.metadata_strategy import MetadataStrategy, get_header_ds
from hisscube.utils.io import get_path_patterns, H5Connector
from hisscube.utils.logging import log_timing, HiSSCubeLogger


class MetadataProcessor:
    def __init__(self, config, photometry, metadata_strategy: MetadataStrategy):
        self.config = config
        self.h5_connector: H5Connector = None
        self.fits_path = None
        self.photometry = photometry
        self.logger = HiSSCubeLogger.logger
        self.metadata_strategy = metadata_strategy

    def reingest_fits_tables(self, h5_connector, image_path, spectra_path, image_pattern=None, spectra_pattern=None):
        image_path_list, spectra_path_list = self.parse_paths(image_path, image_pattern, spectra_path, spectra_pattern)
        self.clean_fits_header_tables(h5_connector)
        image_header_ds, image_header_ds_dtype, spec_header_ds, spec_header_ds_dtype = self.create_fits_header_datasets(
            h5_connector, max_images=len(image_path_list), max_spectra=len(spectra_path_list))
        self.process_fits_headers(h5_connector, image_header_ds, image_header_ds_dtype, image_path, image_path_list)
        self.process_fits_headers(h5_connector, spec_header_ds, spec_header_ds_dtype, spectra_path, spectra_path_list)

    def parse_paths(self, image_path, image_pattern, spectra_path, spectra_pattern):
        image_pattern, spectra_pattern = get_path_patterns(self.config, image_pattern, spectra_pattern)
        image_path_list = get_str_path_list(image_path, image_pattern, self.config.LIMIT_IMAGE_COUNT)
        spectra_path_list = get_str_path_list(spectra_path, spectra_pattern, self.config.LIMIT_SPECTRA_COUNT)
        return image_path_list, spectra_path_list

    def process_fits_headers(self, h5_connector, image_header_ds, image_header_ds_dtype, image_path, image_path_list,
                             offset=0):
        inserted_cnt = self.write_fits_headers(h5_connector, image_header_ds, image_header_ds_dtype, image_path,
                                               image_path_list, offset)
        return inserted_cnt

    def write_fits_headers(self, h5_connector, header_ds, header_ds_dtype, fits_directory_path, path_list, offset=0):
        buf = np.zeros(shape=(self.config.FITS_HEADER_BUF_SIZE,), dtype=header_ds_dtype)
        buf_i = 0
        fits_cnt = 0
        data_type = pathlib.Path(fits_directory_path).name
        for fits_path in tqdm(path_list, desc="Headers for %s" % data_type, position=0, leave=True):
            buf_i, fits_cnt, offset = self._write_fits_header(h5_connector, buf, buf_i, fits_cnt, fits_path, header_ds,
                                                              offset)
        if fits_cnt > 0:
            header_ds.write_direct(buf, source_sel=np.s_[0:buf_i], dest_sel=np.s_[offset:offset + buf_i])
        return fits_cnt

    def create_fits_header_datasets(self, h5_connector, max_images=0, max_spectra=0):
        image_header_ds, image_header_ds_dtype = get_header_ds(h5_connector, max_images,
                                                               self.config.FITS_MAX_PATH_SIZE,
                                                               self.config.FITS_IMAGE_MAX_HEADER_SIZE,
                                                               h5_connector.file, 'fits_images_metadata')
        spec_header_ds, spec_header_ds_dtype = get_header_ds(h5_connector, max_spectra, self.config.FITS_MAX_PATH_SIZE,
                                                             self.config.FITS_SPECTRUM_MAX_HEADER_SIZE,
                                                             h5_connector.file, 'fits_spectra_metadata')
        h5_connector.file.attrs["image_count"] = 0
        h5_connector.file.attrs["spectrum_count"] = 0
        return image_header_ds, image_header_ds_dtype, spec_header_ds, spec_header_ds_dtype

    def clean_fits_header_tables(self, h5_connector):
        if "fits_images_metadata" in h5_connector.file:
            del h5_connector.file["fits_images_metadata"]
        if "fits_spectra_metadata" in h5_connector.file:
            del h5_connector.file["fits_spectra_metadata"]

    @log_timing("fits_headers")
    def _write_fits_header(self, h5_connector, buf, buf_i, fits_cnt, fits_path, header_ds, offset):
        if buf_i >= self.config.FITS_HEADER_BUF_SIZE:
            header_ds.write_direct(buf, source_sel=np.s_[0:buf_i], dest_sel=np.s_[offset:offset + buf_i])
            offset += buf_i
            buf_i = 0
        serialized_header = ujson.dumps(dict(fitsio.read_header(fits_path)))
        buf[buf_i] = (str(fits_path), serialized_header)
        buf_i += 1
        fits_cnt += 1
        h5_connector.fits_total_cnt += 1
        return buf_i, fits_cnt, offset


def get_str_path_list(dir_path, pattern, limit):
    path_generator = [str(x) for x in pathlib.Path(dir_path).rglob(pattern)]
    return list(itertools.islice(path_generator, limit))
