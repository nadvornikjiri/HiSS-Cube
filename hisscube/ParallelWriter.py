import os
import time

import h5py
import logging
from os.path import abspath

import ujson

from hisscube.Writer import Writer
from mpi4py import MPI
import msgpack
import msgpack_numpy as m
from timeit import default_timer as timer

m.patch()

MPI.pickle.__init__(lambda *x: msgpack.dumps(x[0]), msgpack.loads)


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


class ParallelWriter(Writer):

    def __init__(self, h5_file=None, h5_path=None, timings_csv="timings.csv"):
        super().__init__(h5_file, h5_path, timings_csv)
        # mpio
        self.WORK_TAG = 0
        self.KILL_TAG = 1
        self.FINISHED_TAG = 2
        self.comm = MPI.COMM_WORLD
        self.mpi_size = self.comm.Get_size()
        self.mpi_rank = self.comm.Get_rank()
        self.sent_work_cnt = 0
        self.received_work_cnt = 0
        self.received_result_cnt = 0

        logging.basicConfig()
        logging.root.setLevel(self.LOG_LEVEL)
        self.logger = logging.getLogger("rank[%i]" % self.comm.rank)
        mh = MPIFileHandler("logfile.log")
        formatter = logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s',
                                      datefmt='%Y-%m-%d %H:%M:%S')
        mh.setFormatter(formatter)
        self.logger.addHandler(mh)
        self.start_timers = [0] * self.mpi_size
        self.image_batch_cnt = 0
        self.spectrum_batch_cnt = 0
        self.active_workers = 0
        if self.mpi_rank == 0:
            self.logger.info("Rank 0 pid: %d", os.getpid())

    def open_h5_file_parallel(self, truncate=False):
        if truncate and not self.C_BOOSTER:
            if self.MPIO:
                self.f = h5py.File(self.h5_path, 'w', fs_strategy="page", fs_page_size=4096, driver='mpio',
                                   comm=self.comm, libver="latest")
            else:
                self.f = h5py.File(self.h5_path, 'w', fs_strategy="page", fs_page_size=4096, libver="latest")
        else:
            if self.MPIO:
                self.f = h5py.File(self.h5_path, 'r+', driver='mpio',
                                   comm=self.comm, libver="latest")
            else:
                self.f = h5py.File(self.h5_path, 'r+', libver="latest")

    def truncate_h5_file(self):
        self.f = h5py.File(self.h5_path, 'w', libver="latest")
        self.f.close()

    def receive_work(self, status):
        self.wait_for_message(source=0, tag=MPI.ANY_TAG, status=status)
        data = self.comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
        self.logger.debug(
            "Rank %02d: Received work no. %02d from master: %d" % (self.mpi_rank, self.sent_work_cnt, hash(str(data))))
        self.received_work_cnt += 1
        return data

    def send_work(self, batches, dest):
        if len(batches) > 0:
            batch = batches.pop()
            tag = self.WORK_TAG
            self.logger.debug(
                "Send work batch no. %02d to dest %02d: %d " % (self.sent_work_cnt, dest, hash(str(batch))))
            self.comm.send(obj=batch, dest=dest, tag=tag)
            self.sent_work_cnt += 1
            self.start_timers[dest] = timer()

    def wait_for_message(self, source, tag, status):
        while not self.comm.Iprobe(source, tag, status):
            time.sleep(self.POLL_INTERVAL)
        return


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]
