from lib.SDSSCubeParallelWriter import SDSSCubeParallelWriter

H5PATH = "../../SDSS_cube.h5"


class TESTSDSSCubeParallelWriter:
    def test_write_image_metadata(self):
        assert False

    def test_ingest_data(self):
        image_path = "../../data/galaxy_small/images"
        spectra_path = "../../data/galaxy_small/spectra"
        writer = SDSSCubeParallelWriter(H5PATH)
        writer.ingest_data(image_path, spectra_path)
        assert True
