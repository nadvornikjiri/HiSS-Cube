import csv
import os
import time
from hisscube.utils.logging import HiSSCubeLogger
import numpy as np
import re

import msgpack
import msgpack_numpy

msgpack_numpy.patch()

os.environ['MPE_LOGFILE_PREFIX'] = 'ring'
import mpi4py

mpi4py.profile('mpe')

mpi4py.MPI.pickle.__init__(lambda *x: msgpack.dumps(x[0]), msgpack.loads)

rank = mpi4py.MPI.COMM_WORLD.Get_rank()

# import pydevd_pycharm

# port_mapping = [42381, 43567, 35945, 32785, 33119, 42433, 40641, 46823]
# pydevd_pycharm.settrace('localhost', port=port_mapping[rank], stdoutToServer=True, stderrToServer=True)
# print(os.getpid())


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def get_stats(pid):
    stat_vals = list()
    with open("/proc/%s/io" % pid, "r") as stats:
        lines = stats.readlines()
        stat_vals.append(rank)
        for line in lines:
            stat_val = int(re.findall(r'\b\d+\b', line)[0])
            stat_vals.append(stat_val)
    return stat_vals


class MPIHelper:
    size = mpi4py.MPI.COMM_WORLD.Get_size()
    rank = mpi4py.MPI.COMM_WORLD.Get_rank()

    def __init__(self, config):
        self.logger = HiSSCubeLogger.logger
        self.config = config
        self.WORK_TAG = 0
        self.KILL_TAG = 1
        self.FINISHED_TAG = 2
        self.MPI = mpi4py.MPI
        self.comm = mpi4py.MPI.COMM_WORLD
        self.size = self.comm.Get_size()
        self.rank = self.comm.Get_rank()
        self.sent_work_cnt = 0
        self.received_work_cnt = 0
        self.received_result_cnt = 0
        self.active_workers = 0

        if self.rank == 0:
            self.logger.info("Rank 0 pid: %d", os.getpid())

    def receive_work_parsed(self, status):
        image_path_list, offset = None, None
        msg = self.receive_work(status)
        if not status.Get_tag() == self.KILL_TAG:
            image_path_list, offset = msg
        return image_path_list, offset

    def receive_work(self, status):
        self.wait_for_message(source=0, tag=mpi4py.MPI.ANY_TAG, status=status)
        msg = self.comm.recv(source=0, tag=mpi4py.MPI.ANY_TAG, status=status)
        self.logger.debug(
            "Rank %02d: Received work no. %02d from master: %d, tag: %d." % (
            self.rank, self.sent_work_cnt, hash(str(msg)), status.Get_tag()))
        self.received_work_cnt += 1
        return msg

    def send_work(self, batches, dest, offset=0):
        if len(batches) > 0:
            batch = batches.pop()
            msg = (batch, offset)
            tag = self.WORK_TAG
            self.logger.debug(
                "Send work batch no. %02d to dest %02d: %d " % (self.sent_work_cnt, dest, hash(str(batch))))
            self.comm.send(obj=msg, dest=dest, tag=tag)
            self.sent_work_cnt += 1

    def send_work_finished(self, dest):
        tag = self.KILL_TAG
        self.logger.debug("Rank %02d: Terminating worker: %0d" % (self.rank, dest))
        self.comm.send(obj=None, dest=dest, tag=tag)

    def wait_for_message(self, source, tag, status):
        while not self.comm.Iprobe(source, tag, status):
            time.sleep(self.config.POLL_INTERVAL)
        return

    def barrier(self, comm=None, tag=0):
        self.logger.debug("Waiting on barrier.")
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

    def log_proc_stats(self):
        self.barrier()
        self.write_proc_stats()
        self.barrier()

    def write_proc_stats(self):
        buf = np.zeros(8, np.int64)  # 7 stat values + rank
        if rank == 0:
            stats = list()
            stats.append(get_stats(os.getpid()))  # get rank 0 stats
            for i in range(1, self.size):
                self.MPI.COMM_WORLD.Recv(buf, source=i, tag=123)
                stats.append(buf.copy())
        if rank != 0:
            pid = os.getpid()
            buf = np.asarray(get_stats(pid), np.int64)
            self.MPI.COMM_WORLD.Send((buf, 8), dest=0, tag=123)  # sending my worker stats
        if rank == 0:
            with open("proc_stats.txt", "w", newline='') as proc_stat_file:
                proc_stat_writer = csv.writer(proc_stat_file, delimiter=',', quotechar='|',
                                              quoting=csv.QUOTE_MINIMAL)
                proc_stat_writer.writerow(
                    ["Rank", "rchar", "wchar", "syscr", "syscw", "read_bytes", "write_bytes", "cancelled_write_bytes"])

                for stat_line in stats:
                    proc_stat_writer.writerow(stat_line)
