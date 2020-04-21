from os import path

import pandas as pd


def parse_filename(filename):
    _, plate, mjd, fiberid = filename.split("-")
    return int(plate), int(mjd), int(fiberid[:4])


def read_selected_catalog(filepath):
    return pd.read_csv(
        filepath,
        index_col=["plate", "mjd", "fiberid"],
        dtype={
            "platequality": "category",
            "targettype": "category",
            "wavemax": "f4",
            "wavemin": "f4",
            "zwarning": "i4"
            },
        )


def get_filename(plate, mjd, fiberid):
    return "spec-{:04d}-{:05d}-{:04d}.fits".format(plate, mjd, fiberid)


def get_dr_path(plate, mjd, fiberid):
    return path.join("{:04d}".format(plate), get_filename(plate, mjd, fiberid))


def get_spec_filename(spec):
    plate = spec["plate"]
    mjd = spec["mjd"]
    fiberid = spec["fiberid"]
    filename_str = "spec-{:04d}-{:05d}-{:04d}.fits"
    return filename_str.format(plate, mjd, fiberid)


def get_spec_filepath(spec, filename):
    plate = spec["plate"]
    return path.join("{:04d}".format(plate), filename)
