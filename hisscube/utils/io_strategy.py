from abc import ABC, abstractmethod

import ujson


def write_path(header_ds, path, idx):
    header_ds[idx, "path"] = path


def read_path(header_ds, idx):
    return header_ds[idx, "path"]


class IOStrategy(ABC):
    @abstractmethod
    def read_serialized_fits_header(self, ds, idx=0):
        raise NotImplementedError

    @abstractmethod
    def write_serialized_fits_header(self, ds, attrs_dict, idx=0):
        raise NotImplementedError


class SerialTreeIOStrategy(IOStrategy):

    def read_serialized_fits_header(self, ds, idx=0):
        return ujson.loads(ds.attrs["serialized_header"])

    def write_serialized_fits_header(self, ds, attrs_dict, idx=0):
        ds.attrs["serialized_header"] = ujson.dumps(attrs_dict)


def get_orig_header(h5_connector, ds):
    try:
        if ds.attrs["orig_res_link"]:
            orig_image_header = h5_connector.read_serialized_fits_header(h5_connector.file[ds.attrs["orig_res_link"]])
        else:
            orig_image_header = h5_connector.read_serialized_fits_header(ds)
    except KeyError:
        orig_image_header = h5_connector.read_serialized_fits_header(ds)
    return orig_image_header


class CBoostedTreeIOStrategy(IOStrategy):

    def read_serialized_fits_header(self, ds, idx=0):
        return ujson.loads(ds["attrs"]["serialized_header"])

    def write_serialized_fits_header(self, ds, attrs_dict, idx=0):
        if "attrs" not in ds:
            ds["attrs"] = {}
        ds["attrs"]["serialized_header"] = ujson.dumps(attrs_dict)


class SerialDatasetIOStrategy(IOStrategy):

    def read_serialized_fits_header(self, ds, idx=0):
        return ujson.loads(ds[idx, "header"])

    def write_serialized_fits_header(self, ds, attrs_dict, idx=0):
        ds[idx, "header"] = ujson.dumps(attrs_dict)



