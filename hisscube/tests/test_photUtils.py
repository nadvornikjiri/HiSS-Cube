import fitsio
from hisscube import astrometry


def test_get_boundary_coords():
    test_path = "../../data/raw/images/301/4797/1/frame-g-004797-1-0019.fits.bz2"
    with fitsio.FITS(test_path) as f:
        header = f[0].read_header()
        boundaries = astrometry.get_boundary_coords(header)
        print(boundaries)

        assert (all(0 <= coord[0] <= 360 and
                    -90 <= coord[1] <= 90
                    for coord in boundaries))
