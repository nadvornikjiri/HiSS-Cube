from pathlib import Path

import h5py
from tqdm.auto import tqdm

from hisscube.ImageWriter import ImageWriter
from hisscube.VisualizationProcessor import VisualizationProcessor
from hisscube.SpectrumWriter import SpectrumWriter
from timeit import default_timer as timer


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
        start = timer()
        reader = VisualizationProcessor(self.f)
        dense_cube_grp = self.f.require_group(self.DENSE_CUBE_NAME)
        for zoom in range(
                min(self.SPEC_ZOOM_CNT, self.IMG_ZOOM_CNT)):
            spectral_cube = reader.construct_spectral_cube_table(zoom)
            res_grp = dense_cube_grp.require_group(str(zoom))
            visualization = res_grp.require_group("visualization")
            ds = visualization.require_dataset("dense_cube_zoom_%d" % zoom,
                                               spectral_cube.shape,
                                               spectral_cube.dtype,
                                               compression=self.COMPRESSION,
                                               compression_opts=self.COMPRESSION_OPTS,
                                               shuffle=self.SHUFFLE)
            ds.write_direct(spectral_cube)
        end = timer()
        self.logger.info("Dense cube created in: %s", end - start)

    def ingest(self, image_path, spectra_path, image_pattern=None, spectra_pattern=None, truncate_file=None):
        image_pattern, spectra_pattern = self.get_path_patterns(image_pattern, spectra_pattern)
        if self.config.get("Writer", "LIMIT_IMAGE_COUNT"):
            image_paths = list(Path(image_path).rglob(image_pattern))[
                          :self.LIMIT_IMAGE_COUNT]
        else:
            image_paths = list(Path(image_path).rglob(image_pattern))
        if self.LIMIT_SPECTRA_COUNT:
            spectra_paths = list(Path(spectra_path).rglob(spectra_pattern))[
                            :self.LIMIT_SPECTRA_COUNT]
        else:
            spectra_paths = list(Path(spectra_path).rglob(spectra_pattern))
        for image in tqdm(image_paths, desc="Images completed: "):
            self.ingest_image(image)
        for spectrum in tqdm(spectra_paths, desc="Spectra Progress: "):
            self.ingest_spectrum(spectrum)
        if self.CREATE_REFERENCES:
            self.add_image_refs(self.f)
        if self.CREATE_DENSE_CUBE:
            self.create_dense_cube()

    def ingest_metadata(self, no_attrs=False, no_datasets=False):
        self.logger.info("Writing image metadata.")
        self.write_images_metadata(no_attrs, no_datasets)
        self.logger.info("Writing spectra metadata.")
        self.write_spectra_metadata(no_attrs, no_datasets)

    def open_h5_file_serial(self, truncate=False):
        if truncate:
            self.f = h5py.File(self.h5_path, 'w', fs_strategy="page", fs_page_size=4096, libver="latest")
        else:
            self.f = h5py.File(self.h5_path, 'r+', libver="latest")

    def add_region_references(self):
        start = timer()
        self.logger.debug("Adding image region references.")
        self.open_h5_file_serial()
        self.add_image_refs(self.f)
        self.close_h5_file()
        end = timer()
        self.logger.info("Region references added in: %s", end - start)
