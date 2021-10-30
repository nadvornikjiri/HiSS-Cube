# hisscube.py

import os



os.environ['MPE_LOGFILE_PREFIX'] = 'ring'
import mpi4py

mpi4py.profile('mpe')

from mpi4py import MPI

import argparse
from hisscube.WriterFactory import WriterFactory

size = MPI.COMM_WORLD.Get_size()
rank = MPI.COMM_WORLD.Get_rank()

# import pydevd_pycharm
# port_mapping = [38637, 37499]
# pydevd_pycharm.settrace('localhost', port=port_mapping[rank], stdoutToServer=True, stderrToServer=True)

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
writer.ingest_data(fits_image_path, fits_spectra_path, truncate_file=args.truncate)
