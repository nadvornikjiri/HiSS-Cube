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

    @staticmethod
    def get_region_ref(image_ds, cutout_bounds, idx=None):
        if idx is not None:
            return image_ds.regionref[idx, cutout_bounds[0][1][1]:cutout_bounds[1][1][1],
                   cutout_bounds[1][0][0]:cutout_bounds[1][1][0]]
        else:
            return image_ds.regionref[cutout_bounds[0][1][1]:cutout_bounds[1][1][1],
                   cutout_bounds[1][0][0]:cutout_bounds[1][1][0]]

    @staticmethod
    def dereference_region_ref(file, region_ref):
        return file[region_ref][region_ref]

    @staticmethod
    def get_metadata_ref(ds, idx):
        return ds.regionref[idx]


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

    @staticmethod
    def dereference_region_ref(file, region_ref):
        ds = file[region_ref["ds_path"]]
        ds_slice_idx = region_ref["ds_slice_idx"]
        x_min = region_ref["x_min"]
        x_max = region_ref["x_max"]
        y_min = region_ref["y_min"]
        y_max = region_ref["y_max"]
        if x_max == 0 and y_max == 0:  # TODO improve this assertion
            return ds[ds_slice_idx]
        else:
            return ds[ds_slice_idx, x_min:x_max, y_min:y_max, ...]

    @staticmethod
    def get_metadata_ref(ds, idx):
        return ds.name, idx, 0, 0, 0, 0

    @staticmethod
    def get_region_ref(image_ds, cutout_bounds, idx=0):
        return (image_ds.name, idx, cutout_bounds[0][1][1], cutout_bounds[1][1][1], cutout_bounds[1][0][0],
                cutout_bounds[1][1][0])


class ParallelDatasetIOStrategy(SerialDatasetIOStrategy):
    pass



