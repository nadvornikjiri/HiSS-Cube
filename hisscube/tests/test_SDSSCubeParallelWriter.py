import h5py
import pytest

from hisscube.ParallelWriterMWMR import ParallelWriterMWMR

H5PATH = "../../data/processed/SDSS_cube_parallel.h5"


@pytest.fixture(scope="session", autouse=False)
def truncate_test_file(request):
    h5path = H5PATH
    f = h5py.File(h5path, 'w', libver="latest")  # create + truncate file
    f.close()


class TestSDSSCubeParallelWriter:

    @pytest.mark.usefixtures("truncate_test_file")
    def test_ingest_metadata(self):
        image_path = "../../data/raw/galaxy_small/images"
        spectra_path = "../../data/raw/galaxy_small/spectra"
        writer = ParallelWriterMWMR(h5_path=H5PATH)
        writer.open_h5_file_serial()
        writer.ingest_metadata(image_path, spectra_path)
        writer.close_h5_file()
        assert True
