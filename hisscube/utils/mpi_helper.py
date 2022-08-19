import os
import time

import h5py
import logging
from os.path import abspath

import ujson

from hisscube.utils.io import SerialH5Connector
from mpi4py import MPI
import msgpack
import msgpack_numpy as m
from timeit import default_timer as timer

from hisscube.utils.logging import HiSSCubeLogger

m.patch()

MPI.pickle.__init__(lambda *x: msgpack.dumps(x[0]), msgpack.loads)


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


class MPIHelper:

    def __init__(self, config):
        self.logger = HiSSCubeLogger.logger
        self.config = config
        self.WORK_TAG = 0
        self.KILL_TAG = 1
        self.FINISHED_TAG = 2
        self.comm = MPI.COMM_WORLD
        self.mpi_size = self.comm.Get_size()
        self.mpi_rank = self.comm.Get_rank()
        self.sent_work_cnt = 0
        self.received_work_cnt = 0
        self.received_result_cnt = 0
        self.active_workers = 0
        if self.mpi_rank == 0:
            self.logger.info("Rank 0 pid: %d", os.getpid())

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

    def send_work_finished(self, dest):
        tag = self.KILL_TAG
        self.logger.info("Rank %02d: Terminating worker: %0d" % (self.mpi_rank, dest))
        self.comm.send(obj=None, dest=dest, tag=tag)

    def wait_for_message(self, source, tag, status):
        while not self.comm.Iprobe(source, tag, status):
            time.sleep(self.config.POLL_INTERVAL)
        return

    def barrier(self, comm=None, tag=0):
        if not comm:
            comm = self.comm
        sleep_time = self.config.POLL_INTERVAL
        size = comm.Get_size()
        if size == 1:
            return
        rank = comm.Get_rank()
        mask = 1
        while mask < size:
            dst = (rank + mask) % size
            src = (rank - mask + size) % size
            req = comm.isend(None, dst, tag)
            while not comm.Iprobe(src, tag):
                time.sleep(sleep_time)
            comm.recv(None, src, tag)
            req.Wait()
            mask <<= 1
