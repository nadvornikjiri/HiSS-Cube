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


class DataProcessor:
    def __init__(self, config):
        self.config = config


class ImageDataProcessor:
    def __init__(self, config, data_handler):
        self.config = config
        self.data_handler = data_handler
        pass

    def write_datasets(self, res_grp_list, data, file_name):
        img_datasets = []
        for group in res_grp_list:
            res_tuple = group.name.split('/')[-1]
            wanted_res = next(img for img in data if str(tuple(img["zoom"])) == res_tuple)  # parsing 2D resolution
            img_data = np.dstack((wanted_res["flux_mean"], wanted_res["flux_sigma"]))
            img_data[img_data == np.inf] = np.nan
            if self.config.FLOAT_COMPRESS:
                img_data = float_compress(img_data)
            ds = group[file_name]
            ds.write_direct(img_data)
            img_datasets.append(ds)
        return img_datasets


class SpectrumDataProcessor:
    def __init__(self, config, data_handler):
        self.config = config
        self.data_handler = data_handler
        self.data = None

    def write_datasets(self, res_grp_list, data, file_name):
        spec_datasets = []
        for group in res_grp_list:
            res = group.name.split('/')[-1]
            wanted_res = next(spec for spec in data if str(spec["zoom"]) == res)
            spec_data = np.column_stack((wanted_res["wl"], wanted_res["flux_mean"], wanted_res["flux_sigma"]))
            spec_data[spec_data == np.inf] = np.nan
            if self.config.FLOAT_COMPRESS:
                spec_data = float_compress(spec_data)
            ds = group[file_name]
            ds.write_direct(spec_data)
            spec_datasets.append(ds)
        return spec_datasets
