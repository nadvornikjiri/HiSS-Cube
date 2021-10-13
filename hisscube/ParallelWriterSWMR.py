from msgpack import Unpacker

from hisscube.ParallelWriter import ParallelWriter, chunks
from timeit import default_timer as timer
from mpi4py import MPI


class ParallelWriterSWMR(ParallelWriter):
    def __init__(self, h5_file=None, h5_path=None):
        super().__init__(h5_file=h5_file, h5_path=h5_path)
        self.comm_buffer = bytearray(
            self.config.getint("Writer", "BATCH_SIZE") * 100 * 1024 * 1024)  # 100 MBs for one image

    def ingest_data(self, image_path, spectra_path, image_pattern=None, spectra_pattern=None, truncate_file=None):
        image_pattern, spectra_pattern = self.get_path_patterns(image_pattern, spectra_pattern)
        start1 = timer()
        if self.mpi_rank == 0:
            if truncate_file:
                self.truncate_h5_file()
            if not self.f:
                self.open_h5_file_serial()
            self.ingest_metadata(image_path, spectra_path)
        self.comm.Barrier()
        start2 = timer()
        print("Elapsed time: %.2f" % (start2 - start1))
        if self.mpi_rank == 0:
            self.process_parallel(self.image_path_list, "image")
        else:
            self.send_processed_image_data()
        self.comm.Barrier()
        start2 = timer()
        print("Elapsed time: %.2f" % (start2 - start1))
        if self.mpi_rank == 0:
            self.process_parallel(self.spectra_path_list, "spec")
        else:
            self.send_processed_spectra_data()
        self.comm.Barrier()
        start2 = timer()
        print("Elapsed time: %.2f" % (start2 - start1))
        # if self.mpi_rank == 0:
        #     self.logger.info("Rank %02d: Writing image refs." % self.mpi_rank)
        #     self.add_image_refs(self.f)
        #     self.logger.info("Rank %02d: Closing file." % self.mpi_rank)
        #     self.close_h5_file()
        # start2 = timer()
        # print("Elapsed time: %.2f" % (start2 - start1))

    def write_spectrum_data(self, spec_path):
        return super().ingest_spectrum(spec_path)

    def send_processed_image_data(self):
        status = MPI.Status()
        image_path_list = self.receive_work(status)

        while status.Get_tag() != self.KILL_TAG:
            processed_image_batch = []
            for image_path in image_path_list:
                self.logger.info("Rank %02d: Processing image %s." % (self.mpi_rank, image_path))
                self.metadata, self.data = self.cube_utils.get_multiple_resolution_image(image_path,
                                                                                         self.config.getint(
                                                                                             "Handler", "IMG_ZOOM_CNT"))
                self.file_name = image_path.split('/')[-1]

                if "COMMENT" in self.metadata:
                    del self.metadata["COMMENT"]  # TODO fix this serialization hack.
                if "HISTORY" in self.metadata:
                    del self.metadata["HISTORY"]
                processed_image_batch.append(
                    {"metadata": dict(self.metadata), "data": self.data, "file_name": str(self.file_name)})

            self.comm.send(obj=processed_image_batch, dest=0)
            image_path_list = self.receive_work(status)

    def send_processed_spectra_data(self):
        status = MPI.Status()
        spectra_path_list = self.receive_work(status)
        while status.Get_tag() != self.KILL_TAG:
            processed_spectra_batch = []
            for spec_path in spectra_path_list:
                self.logger.info("Rank %02d: Processing spectrum %s." % (self.mpi_rank, spec_path))
                self.metadata, self.data = self.cube_utils.get_multiple_resolution_spectrum(
                    spec_path, self.config.getint("Handler", "SPEC_ZOOM_CNT"),
                    apply_rebin=self.config.getboolean("Preprocessing", "APPLY_REBIN"),
                    rebin_min=self.config.getfloat("Preprocessing", "REBIN_MIN"),
                    rebin_max=self.config.getfloat("Preprocessing", "REBIN_MAX"),
                    rebin_samples=self.config.getint("Preprocessing", "REBIN_SAMPLES"),
                    apply_transmission=self.config.getboolean("Preprocessing", "APPLY_TRANSMISSION_CURVE"))
                self.file_name = spec_path.split('/')[-1]
                if "COMMENT" in self.metadata:
                    del self.metadata["COMMENT"]  # TODO fix this serialization hack.
                if "HISTORY" in self.metadata:
                    del self.metadata["HISTORY"]
                processed_spectra_batch.append(
                    {"metadata": dict(self.metadata), "data": self.data, "file_name": str(self.file_name)})
            self.comm.send(obj=processed_spectra_batch, dest=0)
            spectra_path_list = self.receive_work(status)

    def send_work_finished(self, dest):
        tag = self.KILL_TAG
        self.logger.info("Rank %02d: Terminating worker: %0d" % (self.mpi_rank, dest))
        self.comm.send(obj=None, dest=dest, tag=tag)

    def process_parallel(self, path_list, file_type):
        status = MPI.Status()
        batches = list(chunks(path_list, self.BATCH_SIZE))
        for i in range(1, self.mpi_size):
            self.send_work(batches, dest=i)
        while len(batches) > 0:
            self.comm_buffer = self.comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
            self.logger.info("Rank %02d: Received response from. rank %02d: %d " % (
                self.mpi_rank, status.Get_source(), self.sent_work_cnt))
            self.received_result_cnt += 1
            self.send_work(batches, status.Get_source())
            self.process_response(file_type)
        while self.received_result_cnt < self.sent_work_cnt:
            self.comm_buffer = self.comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
            self.received_result_cnt += 1
            self.process_response(file_type)
        for i in range(1, self.mpi_size):
            self.send_work_finished(dest=i)

    def process_response(self, file_type):

        for response in self.comm_buffer:
            self.metadata = response["metadata"]
            self.data = response["data"]
            self.file_name = response["file_name"]
            if file_type == "spec":
                self.logger.info("Rank %02d: Writing spectrum %s." % (self.mpi_rank, self.file_name))
                self.write_spec_datasets()
            elif file_type == "image":
                self.logger.info("Rank %02d: Writing image %s." % (self.mpi_rank, self.file_name))
                self.write_img_datasets()
