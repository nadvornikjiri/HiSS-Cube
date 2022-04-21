from pathlib import Path

import h5py
from h5py._hl.files import make_fcpl
from tqdm.auto import tqdm

from hisscube.ImageWriter import ImageWriter
from hisscube.VisualizationProcessor import VisualizationProcessor
from hisscube.SpectrumWriter import SpectrumWriter


class Writer(ImageWriter, SpectrumWriter):

    def __init__(self, h5_file=None, h5_path=None, timings_log="timings.csv"):
        super().__init__(h5_file, h5_path, timings_log)

    def create_dense_cube(self):
        """
        Creates the dense cube Group and datasets, needs to be called after the the images and spectra were already
        ingested.
        Returns
        -------

        """
        reader = VisualizationProcessor(self.f)
        dense_cube_grp = self.f.require_group(self.config.get("Handler", "DENSE_CUBE_NAME"))
        for zoom in range(
                min(self.config.getint("Handler", "SPEC_ZOOM_CNT"), self.config.getint("Handler", "IMG_ZOOM_CNT"))):
            spectral_cube = reader.construct_spectral_cube_table(zoom)
            res_grp = dense_cube_grp.require_group(str(zoom))
            visualization = res_grp.require_group("visualization")
            ds = visualization.require_dataset("dense_cube_zoom_%d" % zoom,
                                               spectral_cube.shape,
                                               spectral_cube.dtype,
                                               compression=self.config.get("Writer", "COMPRESSION"),
                                               compression_opts=self.config.get("Writer", "COMPRESSION_OPTS"),
                                               shuffle=self.config.getboolean("Writer", "SHUFFLE"))
            ds.write_direct(spectral_cube)

    def ingest(self, image_path, spectra_path, image_pattern=None, spectra_pattern=None, truncate_file=None):
        image_pattern, spectra_pattern = self.get_path_patterns(image_pattern, spectra_pattern)
        if self.config.get("Writer", "LIMIT_IMAGE_COUNT"):
            image_paths = list(Path(image_path).rglob(image_pattern))[
                          :self.config.getint("Writer", "LIMIT_IMAGE_COUNT")]
        else:
            image_paths = list(Path(image_path).rglob(image_pattern))
        if self.config.get("Writer", "LIMIT_SPECTRA_COUNT"):
            spectra_paths = list(Path(spectra_path).rglob(spectra_pattern))[
                            :self.config.getint("Writer", "LIMIT_SPECTRA_COUNT")]
        else:
            spectra_paths = list(Path(spectra_path).rglob(spectra_pattern))
        for image in tqdm(image_paths, desc="Images completed: "):
            self.ingest_image(image)
        for spectrum in tqdm(spectra_paths, desc="Spectra Progress: "):
            self.ingest_spectrum(spectrum)
        if self.config.getboolean("Writer", "CREATE_REFERENCES"):
            self.add_image_refs(self.f)
        self.close_loggers

    def ingest_metadata(self, image_path, spectra_path, image_pattern=None, spectra_pattern=None, no_attrs=False,
                        no_datasets=False):
        image_pattern, spectra_pattern = self.get_path_patterns(image_pattern, spectra_pattern)
        self.logger.info("Writing image metadata.")
        self.write_images_metadata(image_path, image_pattern, no_attrs, no_datasets)
        self.logger.info("Writing spectra metadata.")
        self.write_spectra_metadata(spectra_path, spectra_pattern, no_attrs, no_datasets)

    def get_path_patterns(self, image_pattern=None, spectra_pattern=None):
        if not image_pattern:
            image_pattern = self.config.get("Writer", "IMAGE_PATTERN")
        if not spectra_pattern:
            spectra_pattern = self.config.get("Writer", "SPECTRA_PATTERN")
        return image_pattern, spectra_pattern

    def open_h5_file_serial(self, truncate=False):
        if truncate:
            self.f = h5py.File(self.h5_path, 'w', fs_strategy="page", fs_page_size=4096, page_buf_size=33554432, libver="latest")
        else:
            self.f = h5py.File(self.h5_path, 'r+', libver="latest")
