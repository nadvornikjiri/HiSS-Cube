import pathlib

import fitsio
import h5py
import healpy
import numpy as np
import ujson

from hisscube.processors.metadata_strategy import MetadataStrategy
from hisscube.utils.astrometry import get_image_lower_res_wcs
from hisscube.utils.io import get_path_patterns, H5Connector, get_orig_header
from hisscube.utils.logging import log_timing, HiSSCubeLogger


class MetadataProcessor:
    def __init__(self, config, photometry, metadata_strategy: MetadataStrategy):
        self.config = config
        self.h5_connector: H5Connector = None
        self.fits_path = None
        self.photometry = photometry
        self.logger = HiSSCubeLogger.logger
        self.metadata_strategy = metadata_strategy

    def require_spatial_grp(self, order, prev, coord):
        return self.metadata_strategy.require

    def add_metadata(self, metadata,  datasets):
        """
        Adds metadata to the HDF5 data sets of the same image or spectrum in multiple resolutions. It also modifies the
        metadata for image where needed and adds the COMMENT and HISTORY attributes as datasets for optimization
        purposes.
        Parameters
        ----------
        datasets    [HDF5 Datasets]

        Returns
        -------

        """
        unicode_dt = h5py.special_dtype(vlen=str)
        image_fits_header = dict(metadata)
        for res_idx, ds in enumerate(datasets):
            if res_idx > 0:
                self.h5_connector.set_attr_ref(ds, "orig_res_link", datasets[0])
                orig_image_fits_header = self.h5_connector.read_serialized_fits_header(datasets[0])
                if self.h5_connector.get_attr(ds, "mime-type") == "image":
                    image_fits_header = get_image_lower_res_wcs(orig_image_fits_header, image_fits_header, res_idx)
            naxis = len(self.h5_connector.get_shape(ds))
            image_fits_header["NAXIS"] = naxis
            for axis in range(naxis):
                image_fits_header["NAXIS%d" % (axis)] = self.h5_connector.get_shape(ds)[axis]
            self.h5_connector.write_serialized_fits_header(ds, image_fits_header)

    def clean_fits_header_tables(self):
        if "fits_images_metadata" in self.h5_connector.file:
            del self.h5_connector.file["fits_images_metadata"]
        if "fits_spectra_metadata" in self.h5_connector.file:
            del self.h5_connector.file["fits_spectra_metadata"]

    def reingest_fits_tables(self, h5_connector, image_path, spectra_path, image_pattern=None, spectra_pattern=None):
        self.h5_connector = h5_connector
        image_pattern, spectra_pattern = get_path_patterns(self.config, image_pattern, spectra_pattern)
        self.clean_fits_header_tables()
        self.create_fits_headers(image_path, image_pattern, spectra_path, spectra_pattern)

    def create_fits_headers(self, image_path, image_pattern, spectra_path, spectra_pattern):
        image_header_ds, image_header_ds_dtype, spec_header_ds, spec_header_ds_dtype = self.create_fits_header_datasets()
        img_cnt = self.write_fits_headers(image_header_ds, image_header_ds_dtype, image_path, image_pattern,
                                          self.config.LIMIT_IMAGE_COUNT)
        self.h5_connector.file.attrs["image_count"] = img_cnt
        spec_cnt = self.write_fits_headers(spec_header_ds, spec_header_ds_dtype, spectra_path, spectra_pattern,
                                           self.config.LIMIT_SPECTRA_COUNT)
        self.h5_connector.file.attrs["spectra_count"] = spec_cnt

    def write_fits_headers(self, header_ds, header_ds_dtype, fits_path, fits_pattern, max_fits_cnt, offset=0):
        buf = np.zeros(shape=(self.config.FITS_HEADER_BUF_SIZE,), dtype=header_ds_dtype)
        buf_i = 0
        fits_cnt = 0
        for fits_path in pathlib.Path(fits_path).rglob(
                fits_pattern):
            buf_i, fits_cnt, offset = self.write_fits_header(buf, buf_i, fits_cnt, fits_path, header_ds, offset)
            if fits_cnt >= max_fits_cnt:
                break
        if fits_cnt > 0:
            header_ds.write_direct(buf, source_sel=np.s_[0:buf_i], dest_sel=np.s_[offset:offset + buf_i])
        return fits_cnt

    @log_timing("fits_headers")
    def write_fits_header(self, buf, buf_i, fits_cnt, fits_path, header_ds, offset):
        if fits_cnt % 100 == 0 and fits_cnt / 100 > 0:
            self.logger.info("Fits cnt: %05d" % fits_cnt)
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

    def create_fits_header_datasets(self):
        max_images = self.config.LIMIT_IMAGE_COUNT
        max_spectra = self.config.LIMIT_SPECTRA_COUNT
        if max_images < 1:
            max_images = self.config.MAX_STORED_IMAGE_HEADERS
        if max_spectra < 1:
            max_spectra = self.config.MAX_STORED_SPECTRA_HEADERS
        path_dtype = h5py.string_dtype(encoding="utf-8", length=self.config.FITS_MAX_PATH_SIZE)
        image_header_dtype = h5py.string_dtype(encoding="utf-8", length=self.config.FITS_IMAGE_MAX_HEADER_SIZE)
        image_header_ds_dtype = [("path", path_dtype), ("header", image_header_dtype)]
        image_header_ds = self.h5_connector.file.create_dataset('fits_images_metadata', (max_images,),
                                                                dtype=image_header_ds_dtype)
        spectrum_header_dtype = h5py.string_dtype(encoding="utf-8", length=self.config.FITS_SPECTRUM_MAX_HEADER_SIZE)
        spec_header_ds_dtype = [("path", path_dtype), ("header", spectrum_header_dtype)]
        spec_header_ds = self.h5_connector.file.create_dataset('fits_spectra_metadata', (max_spectra,),
                                                               dtype=spec_header_ds_dtype)
        self.h5_connector.file.attrs["image_count"] = 0
        self.h5_connector.file.attrs["spectra_count"] = 0
        return image_header_ds, image_header_ds_dtype, spec_header_ds, spec_header_ds_dtype
