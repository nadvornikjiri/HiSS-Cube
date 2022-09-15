from math import log

import h5py
import numpy as np

from hisscube.utils.config import Config


def float_compress(data, ndig=10):
    """
    Makes the data more compressible by zeroing bits of the mantissa.  The method is rewritten from the SDSS IDL
    variant http://www.sdss3.org/dr8/software/idlutils_doc.php#FLOATCOMPRESS.

    This function does not compress the data in an array, but fills
    unnecessary digits of the IEEE floating point representation with
    zeros.  This makes the data more compressible by standard
    compression routines such as compress or gzip.

    The default is to retain 10 binary digits instead of the usual 23
    bits (or 52 bits for double precision), introducing a fractional
    error strictly less than 1/1024).  This is adequate for most
    astronomical images, and results in images that compress a factor
    of 2-4 with gzip.

    Parameters
    ----------
    data    numpy array, type float32 or float64
    ndig    number of binary significant digits to keep

    Returns
    -------

    """
    data = data.astype(np.float32)
    wzer = np.where((data == 0) | (data == np.Inf))

    # replace zeros and infinite values with ones temporarily
    if len(wzer) > 0:
        temp = data[wzer]
        data[wzer] = 1.

    # compute log base 2
    log2 = np.ceil(np.log(np.abs(data)) / log(2.))  # exponent part

    mant = np.round(data / 2.0 ** (log2 - ndig)) / (2.0 ** ndig)  # mantissa, truncated
    out = mant * 2.0 ** log2  # multiple 2^exponent back in

    if len(wzer) > 0:
        out[wzer] = temp

    return out


def get_property_list(config, dataset_shape):
    """
    Creates the property list so it is compatible for parallel file write and reading.

    Parameters
    ----------
    config
    dataset_shape

    Returns
    -------

    """
    dataset_type = h5py.h5t.py_create(np.dtype('f4'))
    dcpl = h5py.h5p.create(h5py.h5p.DATASET_CREATE)
    dcpl.set_alloc_time(h5py.h5d.ALLOC_TIME_EARLY)
    dcpl.set_fill_time(h5py.h5d.FILL_TIME_NEVER)
    space = h5py.h5s.create_simple(dataset_shape)
    if config.COMPRESSION:
        dcpl.set_deflate(config.COMPRESSION_OPTS)
    if config.SHUFFLE:
        dcpl.set_shuffle()
    return dcpl, space, dataset_type


