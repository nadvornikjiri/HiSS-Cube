# hisscube.py
import csv
import os
import re
from time import sleep

import numpy as np

os.environ['MPE_LOGFILE_PREFIX'] = 'ring'
import mpi4py

mpi4py.profile('mpe')

from mpi4py import MPI

import argparse
from hisscube.WriterFactory import WriterFactory

size = MPI.COMM_WORLD.Get_size()
rank = MPI.COMM_WORLD.Get_rank()


# import pydevd_pycharm
# port_mapping = [42053, 42743]
# pydevd_pycharm.settrace('localhost', port=port_mapping[rank], stdoutToServer=True, stderrToServer=True)

def write_proc_stats():
    buf = np.zeros(8, np.int64)  # 7 stat values + rank
    if rank == 0:
        stats = list()
        stats.append(get_stats(os.getpid()))  # get rank 0 stats
        for i in range(1, size):
            MPI.COMM_WORLD.Recv(buf, source=i, tag=123)
            stats.append(buf.copy())
    if rank != 0:
        pid = os.getpid()
        buf = np.asarray(get_stats(pid), np.int64)
        MPI.COMM_WORLD.Send((buf, 8), dest=0, tag=123)  # sending my worker stats
    if rank == 0:
        with open("proc_stats.txt", "w", newline='') as proc_stat_file:
            proc_stat_writer = csv.writer(proc_stat_file, delimiter=',', quotechar='|',
                                          quoting=csv.QUOTE_MINIMAL)
            proc_stat_writer.writerow(
                ["Rank", "rchar", "wchar", "syscr", "syscw", "read_bytes", "write_bytes", "cancelled_write_bytes"])

            for stat_line in stats:
                proc_stat_writer.writerow(stat_line)


def get_stats(pid):
    stat_vals = list()
    with open("/proc/%s/io" % pid, "r") as stats:
        lines = stats.readlines()
        stat_vals.append(rank)
        for line in lines:
            stat_val = int(re.findall(r'\b\d+\b', line)[0])
            stat_vals.append(stat_val)
    return stat_vals


parser = argparse.ArgumentParser(description='Import images and spectra in parallel')
parser.add_argument('input_path', metavar="input", type=str,
                    help="data folder that includes folders images and spectra")
parser.add_argument('output_path', metavar="output", type=str,
                    help="path to HDF5 file, does not need to exist")
parser.add_argument('-t', '--truncate', action='store_const', const=True,
                    help="Should truncate the file if exists?")
args = parser.parse_args()

fits_image_path = "%s/images" % args.input_path
fits_spectra_path = "%s/spectra" % args.input_path

writer = WriterFactory().get_writer(args.output_path)
writer.ingest(fits_image_path, fits_spectra_path, truncate_file=args.truncate)

write_proc_stats()
MPI.COMM_WORLD.barrier()
