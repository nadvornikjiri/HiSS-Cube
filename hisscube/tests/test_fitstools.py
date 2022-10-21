import astropy

from hisscube.utils.fitstools import read_primary_header_quick


def test_read_header_bytes():
    fits_path = "../../data/raw/galaxy_small/images/301/4136/3/frame-g-004136-3-0129.fits"
    with open(fits_path, "rb") as f:
        fits_header = read_primary_header_quick(f)
        with astropy.io.fits.open(fits_path) as f2:
            astropy_header = f2[0].header
            astropy_header.pop("HISTORY", None)
            astropy_header.pop("COMMENT", None)
            assert (len((list(fits_header.keys()))) == len((list(astropy_header.keys()))))
