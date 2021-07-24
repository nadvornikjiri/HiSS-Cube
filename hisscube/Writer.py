from pathlib import Path


from tqdm.auto import tqdm

from hisscube.ImageWriter import ImageWriter
from hisscube.VisualizationProcessor import VisualizationProcessor
from hisscube.SpectrumWriter import SpectrumWriter


class Writer(ImageWriter, SpectrumWriter):

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

    def ingest_metadata(self, image_path, spectra_path):
        self.logger.info("Writing image metadata.")
        self.write_images_metadata(image_path)
        self.logger.info("Writing spectra metadata.")
        self.write_spectra_metadata(spectra_path)

    def ingest_data(self, image_path, spectra_path):
        image_paths = list(Path(image_path).rglob(self.config.get("Writer", "IMAGE_PATTERN")))
        for image in tqdm(image_paths, desc="Images completed: "):
            self.ingest_image(image)
        spectra_paths = list(Path(spectra_path).rglob(self.config.get("Writer", "SPECTRA_PATTERN")))
        for spectrum in tqdm(spectra_paths, desc="Spectra Progress: "):
            self.ingest_spectrum(spectrum)
        self.add_image_refs(self.f)



