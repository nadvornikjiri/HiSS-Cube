import h5py
import ujson
from h5writer import write_hdf5_metadata
import re


from hisscube.ParallelWriterMWMR import ParallelWriterMWMR


class CWriter(ParallelWriterMWMR):

    def __init__(self, h5_file=None, h5_path=None, timings_log="image_timings.csv"):
        super().__init__(h5_file, h5_path, timings_log)
        c_timing_file_name = timings_log.split('/')[-1]
        c_timing_path = "/".join(timings_log.split('/')[:-1]) + "/"
        self.c_timing_log = c_timing_path + "c_" + c_timing_file_name
        self.h5_file_structure = {"name": ""}

    def require_raw_cube_grp(self):
        return self.require_group(self.h5_file_structure, self.config.get("Handler", "ORIG_CUBE_NAME"))

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

    def create_image_h5_dataset(self, group, img_data_shape):
        group["image_dataset"] = {}
        ds = group["image_dataset"]
        ds["name"] = self.file_name
        ds["path"] = "/".join((group["name"], self.file_name))
        ds["shape"] = img_data_shape
        return ds

    def create_spectrum_h5_dataset(self, group, spec_data_shape):
        group["spectrum_dataset"] = {}
        ds = group["spectrum_dataset"]
        ds["name"] = self.file_name
        ds["path"] = "/".join((group["name"], self.file_name))
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
        if not "dataset" in grp:
            grp["dataset"] = {}
            ds = grp["dataset"]
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
        obj["attrs"][key] = obj2["path"]

    def ingest_metadata(self, image_path, spectra_path, image_pattern=None, spectra_pattern=None):
        super().ingest_metadata(image_path, spectra_path, image_pattern, spectra_pattern)
        self.c_write_hdf5_metadata()

    def process_metadata(self, image_path, image_pattern, spectra_path, spectra_pattern, truncate_file):
        image_pattern, spectra_pattern = self.get_path_patterns(image_pattern, spectra_pattern)
        if self.mpi_rank == 0:
            self.logger.info("Writing metadata.")
            self.ingest_metadata(image_path, spectra_path, image_pattern, spectra_pattern)
        self.barrier(self.comm)

    def c_write_hdf5_metadata(self):
        self.logger.info("Initiating C booster for metadata write.")
        write_hdf5_metadata(self.h5_file_structure, self.h5_path, self.c_timing_log)
