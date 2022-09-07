import numpy
import builtins

# temporary measure to shut up astropy's configuration parser.
builtins._ASTROPY_SETUP_ = True
# the following is the original import of pyfits, and only it should be used
from astropy.io import fits as pyfits
from astropy.utils.exceptions import AstropyDeprecationWarning
import warnings

warnings.filterwarnings('ignore', category=AstropyDeprecationWarning)

CARD_SIZE = 80

END_CARD = b'END' + b' ' * (CARD_SIZE - 3)

FITS_BLOCK_SIZE = CARD_SIZE * 36


class FITSError(Exception):
    pass


def read_header_bytes(f, max_header_blocks=80):
    """returns the bytes beloning to a FITS header starting at the current
    position within the file file.

    If the header is not complete after reading maxHeaderBlocks blocks,
    a FITSError is raised.
    """
    if hasattr(f, "encoding"):
        raise IOError("fitstools only works with files opened in binary mode.")
    parts = []

    while True:
        block = f.read(FITS_BLOCK_SIZE)
        if not block:
            raise EOFError('Premature end of file while reading header')

        parts.append(block)
        end_card_pos = block.find(END_CARD)
        if not end_card_pos % CARD_SIZE:
            break

        if len(parts) >= max_header_blocks:
            raise FITSError("No end card found within %d blocks" % max_header_blocks)
    return b"".join(parts)


def read_primary_header_quick(f, max_header_blocks=80):
    """returns a pyfits header for the primary hdu of the opened file file.

    file must be opened in binary mode.

    This is mostly code lifted from pyfits._File._readHDU.  The way
    that class is made, it's hard to use it with stuff from a gzipped
    source, and that's why this function is here.  It is used in the quick
    mode of fits grammars.

    This function is adapted from pyfits.
    """
    fits_header = pyfits.Header.fromstring(
        read_header_bytes(f, max_header_blocks).decode("ascii", "ignore"))
    fits_header.pop("HISTORY", None)    # just discard history, not relevant for machine processing anyway
    fits_header.pop("COMMENT", None)   # just discard comments, not relevant for machine processing anyway
    return fits_header


def read_header_from_path(path):
    with open(path, "rb") as f:
        return read_primary_header_quick(f)
