import h5py

from lib.SDSSCubeParallelWriter import SDSSCubeParallelWriter

H5PATH = "../../SDSS_cube_parallel.h5"


class TestSDSSCubeParallelWriter:
    def test_write_image_metadata(self):
        assert False

    def test_ingest_metadata(self):
        image_path = "../../galaxy_small_decompressed/images"
        spectra_path = "../../data/galaxy_small/spectra"
        writer = SDSSCubeParallelWriter(h5_path=H5PATH)
        writer.ingest_metadata(image_path, spectra_path)
        assert True

    def test_ingest_data(self):
        image_path = "../../galaxy_small_decompressed/images"
        spectra_path = "../../data/galaxy_small/spectra"
        writer = SDSSCubeParallelWriter(h5_path=H5PATH)
        writer.ingest_data(image_path, spectra_path, False)
        assert True
