from os import path

from astropy.time import Time
import pandas as pd


def parse_filename(filename):
    _, lmjd, planid_spid, fiberid = filename.split("-")
    planid, spid = planid_spid.split("_sp")
    return planid, int(lmjd), int(spid), int(fiberid[:3])


def read_general_catalog(catalog_path):
    return pd.read_csv(
	catalog_path,
	dtype={"offsets": "bool"},
	index_col="obsid",
	low_memory=False,
	na_values={
	    'z': -9999.0,
	    "z_err": -9999.0,
	    "snru": -9999.0,
	    "snrg": -9999.0,
	    "snrr": -9999.0,
	    "snri": -9999.0,
	    "snrz": -9999.0,
	    "offset_v": 99.0
	},
	parse_dates=["obsdate"],
	sep='|',
    )


def get_spec_filename(spec):
    lmjd = spec["lmjd"]
    planid = spec["planid"]
    spid = spec["spid"]
    fiberid = spec["fiberid"]
    filename_str = "spec-{}-{}_sp{:02d}-{:03d}.fits.gz"
    return filename_str.format(lmjd, planid, spid, fiberid)


def get_filename(planid, lmjd, spid, fiberid):
    filename_str = "spec-{}-{}_sp{:02d}-{:03}.fits.gz"
    return filename_str.format(lmjd, planid, spid, fiberid)


def get_dr_path(planid, lmjd, spid, fiberid):
    obsdate = Time(lmjd - 1, format="mjd").datetime.strftime("%Y%m%d")
    return path.join(obsdate, planid, get_filename(planid, lmjd, spid, fiberid))


def get_spec_filepath(spec, filename):
    planid = spec["planid"]
    obsdate = spec["obsdate"].strftime("%Y%m%d")
    return path.join(obsdate, planid, filename)
