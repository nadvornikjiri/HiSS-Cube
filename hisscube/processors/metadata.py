import pathlib

import fitsio
import h5py
import numpy as np
import ujson
from tqdm import tqdm

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
        self.h5_connector = h5_connector
        image_pattern, spectra_pattern = get_path_patterns(self.config, image_pattern, spectra_pattern)
        self._clean_fits_header_tables()
        self._create_fits_headers(image_path, image_pattern, spectra_path, spectra_pattern)

    def write_fits_headers(self, header_ds, header_ds_dtype, fits_directory_path, path_list, max_fits_cnt, offset=0):
        buf = np.zeros(shape=(self.config.FITS_HEADER_BUF_SIZE,), dtype=header_ds_dtype)
        buf_i = 0
        fits_cnt = 0
        for fits_path in tqdm(path_list, desc="Writing headers for %s" % fits_directory_path):
            buf_i, fits_cnt, offset = self._write_fits_header(buf, buf_i, fits_cnt, fits_path, header_ds, offset)
            if fits_cnt >= max_fits_cnt:
                break
        if fits_cnt > 0:
            header_ds.write_direct(buf, source_sel=np.s_[0:buf_i], dest_sel=np.s_[offset:offset + buf_i])
        return fits_cnt

    def create_fits_header_datasets(self, max_images=None, max_spectra=None):
        if not max_images:
            max_images = self.config.LIMIT_IMAGE_COUNT
        if not max_spectra:
            max_spectra = self.config.LIMIT_SPECTRA_COUNT
        if not max_images:
            max_images = self.config.MAX_STORED_IMAGE_HEADERS
        if not max_spectra:
            max_spectra = self.config.MAX_STORED_SPECTRA_HEADERS
        image_header_ds, image_header_ds_dtype = get_header_ds(max_images,
                                                               self.config.FITS_MAX_PATH_SIZE,
                                                               self.config.FITS_IMAGE_MAX_HEADER_SIZE,
                                                               self.h5_connector.file, 'fits_images_metadata')
        spec_header_ds, spec_header_ds_dtype = get_header_ds(max_spectra, self.config.FITS_MAX_PATH_SIZE,
                                                             self.config.FITS_SPECTRUM_MAX_HEADER_SIZE,
                                                             self.h5_connector.file, 'fits_spectra_metadata')
        self.h5_connector.file.attrs["image_count"] = 0
        self.h5_connector.file.attrs["spectrum_count"] = 0
        return image_header_ds, image_header_ds_dtype, spec_header_ds, spec_header_ds_dtype

    def _clean_fits_header_tables(self):
        if "fits_images_metadata" in self.h5_connector.file:
            del self.h5_connector.file["fits_images_metadata"]
        if "fits_spectra_metadata" in self.h5_connector.file:
            del self.h5_connector.file["fits_spectra_metadata"]

    def _create_fits_headers(self, image_path, image_pattern, spectra_path, spectra_pattern):
        image_path_list = self.get_path_list(image_path, image_pattern)
        spectra_path_list = self.get_path_list(spectra_path, spectra_pattern)
        image_header_ds, image_header_ds_dtype, spec_header_ds, spec_header_ds_dtype = self.create_fits_header_datasets(
            max_images=len(image_path_list), max_spectra=len(spectra_path_list))
        img_cnt = self.write_fits_headers(image_header_ds, image_header_ds_dtype, image_path, image_path_list,
                                          self.config.LIMIT_IMAGE_COUNT)
        self.h5_connector.file.attrs["image_count"] = img_cnt
        spec_cnt = self.write_fits_headers(spec_header_ds, spec_header_ds_dtype, spectra_path, spectra_path_list,
                                           self.config.LIMIT_SPECTRA_COUNT)
        self.h5_connector.file.attrs["spectrum_count"] = spec_cnt

    @staticmethod
    def get_path_list(fits_directory_path, fits_pattern):
        return list(pathlib.Path(fits_directory_path).rglob(fits_pattern))

    @log_timing("fits_headers")
    def _write_fits_header(self, buf, buf_i, fits_cnt, fits_path, header_ds, offset):
        if fits_cnt % 100 == 0 and fits_cnt / 100 > 0:
            self.logger.info("Fits idx: %05d" % fits_cnt)
        if buf_i >= self.config.FITS_HEADER_BUF_SIZE:
            header_ds.write_direct(buf, source_sel=np.s_[0:buf_i], dest_sel=np.s_[offset:offset + buf_i])
            offset += buf_i
            buf_i = 0
        serialized_header = ujson.dumps(dict(fitsio.read_header(fits_path)))
        buf[buf_i] = (str(fits_path), serialized_header)
        buf_i += 1
        fits_cnt += 1
        self.h5_connector.fits_total_cnt += 1
        return buf_i, fits_cnt, offset
