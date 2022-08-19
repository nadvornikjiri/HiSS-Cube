import cProfile
import logging
import csv
import time
import os
from pathlib import Path
import numpy as np
import csv
import re

from mpi4py import MPI

from hisscube.utils.config import Config

rank = MPI.COMM_WORLD.Get_rank()
size = MPI.COMM_WORLD.Get_size()


class MPIFileHandler(logging.FileHandler):
    def __init__(self,
                 filename,
                 mode=MPI.MODE_WRONLY | MPI.MODE_CREATE | MPI.MODE_APPEND,
                 encoding='utf-8',
                 delay=False,
                 comm=MPI.COMM_WORLD):
        self.baseFilename = abspath(filename)
        self.mode = mode
        self.encoding = encoding
        self.comm = comm
        if delay:
            # We don't open the stream, but we still need to call the
            # Handler constructor to set level, formatter, lock etc.
            logging.Handler.__init__(self)
            self.stream = None
        else:
            logging.StreamHandler.__init__(self, self._open())

    def _open(self):
        stream = MPI.File.Open(self.comm, self.baseFilename, self.mode)
        stream.Set_atomicity(True)
        return stream

    def emit(self, record):
        """
        Emit a record.
        If a formatter is specified, it is used to format the record.
        The record is then written to the stream with a trailing newline.  If
        exception information is present, it is formatted using
        traceback.print_exception and appended to the stream.  If the stream
        has an 'encoding' attribute, it is used to determine how to do the
        output to the stream.

        Modification:
            stream is MPI.File, so it must use `Write_shared` method rather
            than `write` method. And `Write_shared` method only accept
            bytestring, so `encode` is used. `Write_shared` should be invoked
            only once in each all of this emit function to keep atomicity.
        """
        try:
            msg = self.format(record)
            stream = self.stream
            stream.Write_shared((msg + self.terminator).encode(self.encoding))
            # self.flush()
        except Exception:
            self.handleError(record)

    def close(self):
        if self.stream:
            self.stream.Sync()
            self.stream.Close()
            self.stream = None


class TimingsCSVLogger:
    def __init__(self, path):
        super().__init__()
        self.file = open(path, "w", newline='')
        self.logger = csv.writer(self.file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

    def __del__(self):
        self.file.close()

    def log_timing(self, values):
        self.logger.writerow(values)
        self.file.flush()

def measured_time():
    times = os.times()
    # return times.elapsed - (times.system + times.user)
    return times.elapsed


def profile(filename=None, comm=MPI.COMM_WORLD):
    def prof_decorator(f):
        def wrap_f(*args, **kwargs):
            pr = cProfile.Profile(measured_time)
            pr.enable()
            result = f(*args, **kwargs)
            pr.disable()

            if filename is None:
                pr.print_stats()
            else:
                filename_r = filename + ".{}".format(comm.rank)
                pr.dump_stats(filename_r)

            return result

        return wrap_f

    return prof_decorator


def log_timing(prefix, timings_path="logs/timings.csv"):
    os.makedirs(os.path.dirname(timings_path), exist_ok=True)  # create parent dirs if not exists
    prefix = "%s_%d" % (prefix, rank)  # adding support for parallel logging
    logger = get_timings_logger(timings_path, prefix)

    def middle(func):
        def wrapper(self, *args, **kwargs):
            start_time = time.time()
            result = func(self, *args, **kwargs)
            elapsed_time = time.time() - start_time
            logger.log_timing([self.metadata_processor.fits_total_cnt, self.metadata_processor.grp_cnt, elapsed_time])
            return result

        return wrapper

    return middle


def get_timings_logger(timings_path, prefix):
    timing_file_name = timings_path.split('/')[-1]
    timing_path = "/".join(timings_path.split('/')[:-1])
    if timing_path != "":
        timing_path += "/"
    timing_log_path = "%s%s_%s" % (timing_path, prefix, timing_file_name)
    logger = TimingsCSVLogger(timing_log_path)
    logger.log_timing(["Image/Spectrum count", "Group count", "Time"])
    return logger


def get_stats(pid):
    stat_vals = list()
    with open("/proc/%s/io" % pid, "r") as stats:
        lines = stats.readlines()
        stat_vals.append(rank)
        for line in lines:
            stat_val = int(re.findall(r'\b\d+\b', line)[0])
            stat_vals.append(stat_val)
    return stat_vals


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


def log_proc_stats():
    MPI.barrier(MPI.COMM_WORLD)
    write_proc_stats()
    MPI.barrier(MPI.COMM_WORLD)


def get_application_logger():
    config = Config()
    logging.basicConfig()
    logging.root.setLevel(config.LOG_LEVEL)
    logger = logging.getLogger("rank[%i]" % rank)
    mh = MPIFileHandler("logfile.log")
    formatter = logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    mh.setFormatter(formatter)
    logger.addHandler(mh)
    return logger


class HiSSCubeLogger:
    logger = get_application_logger()


def get_c_timings_path(timings_log="logs/timings.csv"):
    c_timing_file_name = timings_log.split('/')[-1]
    c_timing_path = "/".join(timings_log.split('/')[:-1])
    if c_timing_path != "":
        c_timing_path += "/"
    return c_timing_path + "c_" + c_timing_file_name
