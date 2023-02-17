from abc import ABC, abstractmethod
from collections import deque
from datetime import datetime
from itertools import chain
from sys import getsizeof, stderr

import h5py
import mpi4py.MPI
import numpy as np
import pandas as pd
from astropy.time import Time
from h5py import h5f
from h5py.h5p import PropFAID

from hisscube.processors.data import get_property_list
from hisscube.utils.config import Config
from hisscube.utils.io_strategy import IOStrategy
from hisscube.utils.logging import get_c_timings_path
from hisscube.utils.nexus import set_nx_entry

size = mpi4py.MPI.COMM_WORLD.Get_size()
rank = mpi4py.MPI.COMM_WORLD.Get_rank()


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


def get_path_patterns(config, image_pattern=None, spectra_pattern=None):
    if not image_pattern:
        image_pattern = config.IMAGE_PATTERN
    if not spectra_pattern:
        spectra_pattern = config.SPECTRA_PATTERN
    return image_pattern, spectra_pattern


def truncate(h5_path, config=None):
    if config and config.USE_SUBFILING:
        f = h5py.File(h5_path, 'w', driver='mpio', comm=mpi4py.MPI.COMM_WORLD, libver="latest",
                      ioc_thread_pool_size=config.IOC_THREADPOOL_SIZE,
                      ioc_selection=config.IOC_SELECTION,
                      stripe_size=config.STRIPE_SIZE,
                      stripe_count=config.STRIPE_COUNT)
        f.close()
    elif rank == 0:
        f = h5py.File(h5_path, 'w', fs_strategy="page", fs_page_size=4096, libver="latest")
        f.close()


class H5Connector(ABC):
    def __init__(self, h5_path, config: Config, io_strategy: IOStrategy = None, mpi_comm=None):
        self.h5_path = h5_path
        self.config = config
        self.comm = mpi_comm
        self.strategy = io_strategy
        self.grp_cnt = 0
        self.fits_total_cnt = 0
        self.file = None

    def __enter__(self):
        self.open_h5_file()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close_h5_file()

    @abstractmethod
    def open_h5_file(self, truncate_file=False):
        raise NotImplementedError

    def close_h5_file(self):
        self.file.close()

    def read_serialized_fits_header(self, ds, idx=0):
        return self.strategy.read_serialized_fits_header(ds, idx)

    def write_serialized_fits_header(self, ds, attrs_dict, idx=0):
        return self.strategy.write_serialized_fits_header(ds, attrs_dict, idx)

    def require_semi_sparse_cube_grp(self):
        grp = self.require_group(self.file, self.config.SPARSE_CUBE_NAME)
        set_nx_entry(grp, self)
        return grp

    def require_dense_group(self):
        grp = self.require_group(self.file, self.config.DENSE_CUBE_NAME)
        set_nx_entry(grp, self)
        return grp

    def get_dense_group(self):
        return self.file[self.config.DENSE_CUBE_NAME]

    def require_group(self, parent_grp, name, track_order=False):
        if not name in parent_grp:
            self.grp_cnt += 1
            return parent_grp.create_group(name, track_order=track_order)
        grp = parent_grp[name]
        return grp

    def create_image_h5_dataset(self, group, file_name, img_data_shape, chunk_size=None):
        return self.create_dataset(group, file_name, img_data_shape, chunk_size)

    def create_dataset(self, group, dataset_name, dataset_shape, chunk_size=None, dataset_type=None):
        if not dataset_type:
            dataset_type = h5py.h5t.py_create(np.dtype('<f4'))
        else:
            dataset_type = h5py.h5t.py_create(dataset_type)
        ds_name = dataset_name.encode('utf-8')
        if not ds_name in group:
            dcpl, space = get_property_list(self.config, dataset_shape)
            if chunk_size and not (any(dataset_shape) == 0):
                dcpl.set_chunk(chunk_size)
            dsid = h5py.h5d.create(group.id, ds_name, dataset_type, space, dcpl=dcpl)
            ds = h5py.Dataset(dsid)
        else:
            ds = group[ds_name]
        return ds

    def recreate_regionref_dataset(self, ds_name, item_count, spec_zoom_group, dtype=None, wl_count=None):
        if ds_name in spec_zoom_group:
            del spec_zoom_group[ds_name]
        if self.config.LINK_BATCH_SIZE > item_count:
            chunk_size = item_count
        else:
            chunk_size = self.config.LINK_BATCH_SIZE
        if wl_count:
            ds_shape = (item_count, wl_count, self.config.MAX_CUTOUT_REFS)
            chunk_shape = (chunk_size, wl_count, self.config.MAX_CUTOUT_REFS)
        else:
            ds_shape = (item_count, self.config.MAX_CUTOUT_REFS)
            chunk_shape = (chunk_size, self.config.MAX_CUTOUT_REFS)
        return self.create_dataset(spec_zoom_group, ds_name,
                                   ds_shape,
                                   chunk_size=chunk_shape,
                                   dataset_type=dtype)

    def create_spectrum_h5_dataset(self, group, file_name, spec_data_shape, chunk_size=None):
        return self.create_dataset(group, file_name, spec_data_shape, chunk_size)

    def set_spectrum_count(self, count):
        self.set_attr(self.file, "spectrum_count", count)

    def set_image_count(self, count):
        self.set_attr(self.file, "image_count", count)

    def get_spectrum_count(self):
        return self.get_attr(self.file, "spectrum_count")

    def get_image_count(self):
        return self.get_attr(self.file, "image_count")

    def set_target_count(self, count):
        self.set_attr(self.file[self.config.DENSE_CUBE_NAME], "target_count", count)

    def get_target_count(self):
        return self.get_attr(self.file[self.config.DENSE_CUBE_NAME], "target_count")

    @staticmethod
    def get_name(grp):
        return grp.name

    @staticmethod
    def set_attr(obj, key, val):
        obj.attrs[key] = val

    @staticmethod
    def set_attr_ref(obj, key, obj2):
        obj.attrs[key] = obj2.ref

    @staticmethod
    def get_attr(ds, key):
        return ds.attrs[key]

    @staticmethod
    def require_dataset(grp, name, shape, dtype):
        return grp.require_dataset(name, shape, dtype)

    @staticmethod
    def get_shape(ds):
        return ds.shape

    def get_region_ref(self, image_ds, cutout_bounds, idx=None):
        return self.strategy.get_region_ref(image_ds, cutout_bounds, idx)

    def get_metadata_ref(self, ds, idx=0):
        return self.strategy.get_metadata_ref(ds, idx)

    def dereference_region_ref(self, reg_ref):
        return self.strategy.dereference_region_ref(self.file, reg_ref)


class SerialH5Writer(H5Connector):

    def __init__(self, h5_path, config, io_strategy: IOStrategy, mpi_comm):
        super().__init__(h5_path, config, io_strategy, mpi_comm)

    def open_h5_file(self, truncate_file=False):
        try:
            if not self.config.USE_SUBFILING:
                self.file = h5py.File(self.h5_path, 'r+', libver="latest")
            else:
                self.file = h5py.File(self.h5_path, 'r+', driver="mpio", comm=self.comm, libver="latest",
                                      ioc_thread_pool_size=self.config.IOC_THREADPOOL_SIZE,
                                      ioc_selection=self.config.IOC_SELECTION,
                                      stripe_size=self.config.STRIPE_SIZE,
                                      stripe_count=self.config.STRIPE_COUNT)

        except FileNotFoundError:
            truncate(self.h5_path, self.config)
            self.open_h5_file()


class SerialH5Reader(H5Connector):

    def __init__(self, h5_path, config, io_strategy: IOStrategy):
        super().__init__(h5_path, config, io_strategy)

    def open_h5_file(self, truncate_file=False):
        self.file = h5py.File(self.h5_path, 'r', libver="latest")


class H5ReaderSWMR(H5Connector):
    def __init__(self, h5_path, config: Config, io_strategy: IOStrategy):
        super().__init__(h5_path, config, io_strategy)

    def open_h5_file(self, truncate_file=False):
        self.file = h5py.File(self.h5_path, 'r', libver="latest", swmr=True)


class H5WriterSWMR(H5Connector):
    def __init__(self, h5_path, config: Config, io_strategy: IOStrategy):
        super().__init__(h5_path, config, io_strategy)

    def open_h5_file(self, truncate_file=False):
        self.file = h5py.File(self.h5_path, 'r+', libver="latest", swmr=True)


class ParallelH5Writer(H5Connector):
    def __init__(self, h5_path, config, io_strategy: IOStrategy, mpi_comm=None):
        super().__init__(h5_path, config, io_strategy)
        self.comm = mpi_comm
        if not self.comm:
            self.comm = mpi4py.MPI.COMM_WORLD

    def open_h5_file(self, truncate_file=False):
        if truncate_file:
            truncate(self.h5_path, self.config)
        if not self.config.USE_SUBFILING:
            self.file = h5py.File(self.h5_path, 'r+', driver='mpio',
                                  comm=self.comm, libver="latest")
        else:
            self.file = h5py.File(self.h5_path, 'r+', driver='mpio', comm=self.comm, libver="latest",
                                  ioc_thread_pool_size=self.config.IOC_THREADPOOL_SIZE,
                                  ioc_selection=self.config.IOC_SELECTION,
                                  stripe_size=self.config.STRIPE_SIZE,
                                  stripe_count=self.config.STRIPE_COUNT)


class CBoostedMetadataBuildWriter(SerialH5Writer):

    def __init__(self, h5_path, config, io_strategy: IOStrategy):
        super().__init__(h5_path, config, io_strategy, None)
        self.c_timing_log = get_c_timings_path()
        self.h5_file_structure = {"name": ""}

    def require_semi_sparse_cube_grp(self):
        grp = self.require_group(self.h5_file_structure, self.config.SPARSE_CUBE_NAME)
        set_nx_entry(grp, self)
        return grp

    def require_dense_group(self):
        grp = self.require_group(self.h5_file_structure, self.config.DENSE_CUBE_NAME)
        set_nx_entry(grp, self)
        return grp

    def require_group(self, parent_grp, name, track_order=False):
        if not name in parent_grp:
            self.grp_cnt += 1
            parent_grp[name] = {}
            parent_grp[name]["name"] = "/".join((parent_grp["name"], name))
        child_grp = parent_grp[name]
        if track_order:
            child_grp["track_order"] = True
        return child_grp

    def set_attr(self, obj, key, val):

        if obj is None or isinstance(obj, h5py._hl.files.File):
            obj = self.h5_file_structure
        if "attrs" not in obj:
            obj["attrs"] = {}
        obj["attrs"][key] = val

    @staticmethod
    def get_attr(obj, key):
        return obj["attrs"][key]

    def create_image_h5_dataset(self, group, file_name, img_data_shape, chunk_size=None):
        group["image_dataset"] = {}
        ds = group["image_dataset"]
        ds["name"] = file_name
        ds["path"] = "/".join((group["name"], file_name))
        ds["shape"] = img_data_shape
        return ds

    def create_spectrum_h5_dataset(self, group, file_name, spec_data_shape, chunk_size=None):
        group["spectrum_dataset"] = {}
        ds = group["spectrum_dataset"]
        ds["name"] = file_name
        ds["path"] = "/".join((group["name"], file_name))
        ds["shape"] = spec_data_shape
        return ds

    @staticmethod
    def require_dataset(grp, name, shape, dtype):
        if not name in grp:
            grp[name] = {}
            ds = grp[name]
            ds["name"] = name
            ds["shape"] = shape
            if dtype == h5py.regionref_dtype:
                ds["dtype"] = "regionref"

    @staticmethod
    def get_name(grp):
        return grp["name"]

    @staticmethod
    def get_shape(ds):
        return ds["shape"]

    @staticmethod
    def set_attr_ref(obj, key, obj2):
        obj["attrs"][key] = obj2["path"]  # the obj2["path"] is not needed ATM.


class PandasHDFWriter(H5Connector):

    def open_h5_file(self, truncate_file=False):
        self.file = pd.HDFStore(self.h5_path)


def get_image_header_dataset(h5_connector):
    return h5_connector.file["fits_images_metadata"]


def get_spectrum_header_dataset(h5_connector):
    return h5_connector.file["fits_spectra_metadata"]


def get_fits_path(metadata_ds, cnt):
    return metadata_ds[cnt]["path"]


def get_str_paths(ds):
    paths = ds[:]["path"]
    for path in paths:
        if path:
            path = path.decode('utf-8')
            yield path
        else:
            break


def get_spectra_str_paths(h5_connector):
    spec_path_ds = get_spectrum_header_dataset(h5_connector)
    spec_path_list = list(get_str_paths(spec_path_ds))
    return spec_path_list


def get_image_str_paths(h5_connector):
    image_path_ds = get_image_header_dataset(h5_connector)
    image_path_list = list(get_str_paths(image_path_ds))
    return image_path_list


def get_time_from_image(orig_image_header):
    time_attr = orig_image_header["DATE-OBS"]
    try:
        time = Time(time_attr, format='isot', scale='tai').mjd
    except ValueError:
        time = Time(datetime.strptime(time_attr, "%d/%m/%y")).mjd
    return time


def get_error_ds(h5_connector, ds):
    return h5_connector.file[ds.attrs["error_ds"]]
