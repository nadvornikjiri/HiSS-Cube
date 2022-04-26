import configparser
import pathlib

from hisscube.CWriter import CWriter
from hisscube.Writer import Writer
from hisscube.ParallelWriterMWMR import ParallelWriterMWMR
from hisscube.ParallelWriterSWMR import ParallelWriterSWMR


class WriterFactory:
    def __init__(self):
        lib_path = pathlib.Path(__file__).parent.absolute()
        self.config = configparser.ConfigParser(allow_no_value=True)
        self.config.read("%s/config.ini" % lib_path)

    def get_writer(self, h5_path):
        if not self.MPIO:
            return Writer(h5_path=h5_path)
        else:
            if self.PARALLEL_MODE == "SWMR":
                return ParallelWriterSWMR(h5_path=h5_path)
            elif self.PARALLEL_MODE == "MWMR":
                if self.C_BOOSTER:
                    return CWriter(h5_path=h5_path)
                else:
                    return ParallelWriterMWMR(h5_path=h5_path)
            else:
                raise Exception("Unsupported parallel writer mode.")
