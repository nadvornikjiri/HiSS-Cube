from __future__ import print_function
from sys import getsizeof, stderr
from itertools import chain
from collections import deque
import pytest
from tqdm.auto import tqdm

import h5py

try:
    from reprlib import repr
except ImportError:
    pass

from hisscube.CWriter import CWriter

FITS_IMAGE_PATH = "../../data/raw/galaxy_small/images"
FITS_SPECTRA_PATH = "../../data/raw/galaxy_small/spectra"
H5_PATH = "../../results/SDSS_cube_c_par.h5"


class TestCWriter:
    def test_write_images_metadata(self):
        writer = CWriter()
        image_pattern, spectra_pattern = writer.get_path_patterns()
        writer.write_images_metadata(FITS_IMAGE_PATH, image_pattern, no_attrs=False, no_datasets=False)
        assert True

    def test_write_spectra_metadata(self):
        writer = CWriter()
        image_pattern, spectra_pattern = writer.get_path_patterns()
        writer.write_spectra_metadata(FITS_SPECTRA_PATH, spectra_pattern)
        assert True

    def test_process_metadata(self):
        writer = CWriter(h5_path=H5_PATH, timings_log="logs/test_log.csv")
        image_pattern, spectra_pattern = writer.get_path_patterns()
        writer.process_metadata(FITS_IMAGE_PATH, image_pattern, FITS_SPECTRA_PATH, spectra_pattern, truncate_file=True,
                                no_datasets=False, no_attrs=False)

        h5_file = h5py.File(H5_PATH, libver="latest")
        test_ds = h5_file[
            "/semi_sparse_cube/5/22/90/362/1450/5802/23208/92832/4604806771.19/3551/(1024, 744)/frame-u-004899-2-0260.fits"]
        orig_res_link = test_ds.attrs["orig_res_link"]
        orig_res_ds = h5_file[
            "/semi_sparse_cube/5/22/90/362/1450/5802/23208/92832/4604806771.19/3551/(2048, 1489)/frame-u-004899-2-0260.fits"]
        orig_res_ds_name = orig_res_ds.name.split('/')[-1]
        test_ds_name = h5_file[orig_res_link].name.split('/')[-1]
        assert (orig_res_ds_name == test_ds_name)

    def test_add_spec_refs_multiple(self):
        writer = CWriter(h5_path=H5_PATH, timings_log="logs/test_log.csv")
        writer.add_region_references()
        assert True


def total_size(o, handlers={}, verbose=False):
    """ Returns the approximate memory footprint an object and all of its contents.

    Automatically finds the contents of the following builtin containers and
    their subclasses:  tuple, list, deque, dict, set and frozenset.
    To search other containers, add handlers to iterate over their contents:

        handlers = {SomeContainerClass: iter,
                    OtherContainerClass: OtherContainerClass.get_elements}

    """
    dict_handler = lambda d: chain.from_iterable(d.items())
    all_handlers = {tuple: iter,
                    list: iter,
                    deque: iter,
                    dict: dict_handler,
                    set: iter,
                    frozenset: iter,
                    }
    all_handlers.update(handlers)  # user handlers take precedence
    seen = set()  # track which object id's have already been seen
    default_size = getsizeof(0)  # estimate sizeof object without __sizeof__

    def sizeof(o):
        if id(o) in seen:  # do not double count the same object
            return 0
        seen.add(id(o))
        s = getsizeof(o, default_size)

        if verbose:
            print(s, type(o), repr(o), file=stderr)

        for typ, handler in all_handlers.items():
            if isinstance(o, typ):
                s += sum(map(sizeof, handler(o)))
                break
        return s

    return sizeof(o)
