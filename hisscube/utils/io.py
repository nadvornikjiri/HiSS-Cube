from abc import ABC, abstractmethod
from ast import literal_eval as make_tuple, literal_eval
from collections import deque
from itertools import chain
from sys import getsizeof, stderr
from timeit import default_timer as timer

import h5py
import mpi4py.MPI
import ujson
from h5writer import write_hdf5_metadata

from hisscube.processors.data import get_property_list
from hisscube.utils.logging import get_c_timings_path


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


def truncate(h5_path):
    f = h5py.File(h5_path, 'w', libver="latest")
    f.close()


class FITSHandler:
    pass


class H5Connector(ABC):
    def __init__(self, h5_path, config):
        self.h5_path = h5_path
        self.config = config
        self.grp_cnt = 0
        self.f = None

    def __enter__(self):
        self.open_h5_file()

    def __exit__(self):
        self.close_h5_file()

    @abstractmethod
    def open_h5_file(self, truncate_file=False):
        raise NotImplementedError

    def close_h5_file(self):
        self.f.close()

    def require_raw_cube_grp(self):
        return self.require_group(self.f, self.config.ORIG_CUBE_NAME)

    def require_group(self, parent_grp, name, track_order=False):
        if not name in parent_grp:
            self.grp_cnt += 1
            return parent_grp.create_group(name, track_order=track_order)
        grp = parent_grp[name]
        return grp

    def create_image_h5_dataset(self, group, file_name, img_data_shape):
        dcpl, space, img_data_dtype = get_property_list(self.config, img_data_shape)
        if self.config.CHUNK_SIZE:
            dcpl.set_chunk(literal_eval(self.config.CHUNK_SIZE))
        dsid = h5py.h5d.create(group.id, file_name.encode('utf-8'), img_data_dtype, space,
                               dcpl=dcpl)
        ds = h5py.Dataset(dsid)
        return ds

    def create_spectrum_h5_dataset(self, group, file_name, spec_data_shape):
        dcpl, space, spec_data_dtype = get_property_list(self.config, spec_data_shape)
        if not file_name in group:
            dsid = h5py.h5d.create(group.id, file_name.encode('utf-8'), spec_data_dtype, space, dcpl=dcpl)
            ds = h5py.Dataset(dsid)
        else:
            ds = group[file_name]
        return ds

    @staticmethod
    def get_name(self, grp):
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
    def read_serialized_fits_header(ds):
        return ujson.loads(ds.attrs["serialized_header"])

    @staticmethod
    def write_serialized_fits_header(ds, attrs_dict):
        ds.attrs["serialized_header"] = ujson.dumps(attrs_dict)

    @staticmethod
    def require_dataset(grp, name, shape, dtype):
        grp.require_dataset(name, shape, dtype)

    @staticmethod
    def get_shape(ds):
        return ds.shape


class SerialH5Connector(H5Connector):

    def __init__(self, h5_path):
        super().__init__(h5_path)

    def open_h5_file(self, truncate_file=False):
        if truncate_file:
            self.f = h5py.File(self.h5_path, 'w', fs_strategy="page", fs_page_size=4096, libver="latest")
        else:
            self.f = h5py.File(self.h5_path, 'r+', libver="latest")


class ParallelH5Connector(H5Connector):
    def __init__(self, h5_path, mpi_comm=None):
        super().__init__(h5_path)
        self.comm = mpi_comm
        if not self.comm:
            self.comm = mpi4py.MPI.COMM_WORLD

    def open_h5_file(self, truncate_file=False):
        if truncate_file and not self.config.C_BOOSTER:
            self.f = h5py.File(self.h5_path, 'w', fs_strategy="page", fs_page_size=4096, driver='mpio',
                               comm=self.comm, libver="latest")
        else:
            self.f = h5py.File(self.h5_path, 'r+', driver='mpio',
                               comm=self.comm, libver="latest")


class CBoostedMetadataBuildConnector(SerialH5Connector):

    def __init__(self, h5_path):
        super().__init__(h5_path)
        self.c_timing_log = get_c_timings_path()
        self.h5_file_structure = {"name": ""}

    def require_raw_cube_grp(self):
        return self.require_group(self.h5_file_structure, self.config.ORIG_CUBE_NAME)

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
        if obj is None:
            obj = self.h5_file_structure
        if "attrs" not in obj:
            obj["attrs"] = {}
        obj["attrs"][key] = val

    @staticmethod
    def get_attr(obj, key):
        return obj["attrs"][key]

    def create_image_h5_dataset(self, group, file_name, img_data_shape):
        group["image_dataset"] = {}
        ds = group["image_dataset"]
        ds["name"] = file_name
        ds["path"] = "/".join((group["name"], file_name))
        ds["shape"] = img_data_shape
        return ds

    def create_spectrum_h5_dataset(self, group, file_name, spec_data_shape):
        group["spectrum_dataset"] = {}
        ds = group["spectrum_dataset"]
        ds["name"] = file_name
        ds["path"] = "/".join((group["name"], file_name))
        ds["shape"] = spec_data_shape
        return ds

    @staticmethod
    def write_serialized_fits_header(ds, attrs_dict):
        if "attrs" not in ds:
            ds["attrs"] = {}
        ds["attrs"]["serialized_header"] = ujson.dumps(attrs_dict)

    @staticmethod
    def read_serialized_fits_header(ds):
        return ujson.loads(ds["attrs"]["serialized_header"])

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


