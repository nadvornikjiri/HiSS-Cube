# hisscube.py
import os
import argparse
from mpi4py import MPI

from hisscube.dependency_injector import HiSSCubeProvider
from hisscube.command import CLICommandInvoker
from hisscube.utils.logging import log_proc_stats

os.environ['MPE_LOGFILE_PREFIX'] = 'ring'
import mpi4py

mpi4py.profile('mpe')

size = mpi4py.MPI.COMM_WORLD.Get_size()
rank = mpi4py.MPI.COMM_WORLD.Get_rank()

# import pydevd_pycharm
# port_mapping = [36791, 39053]
# pydevd_pycharm.settrace('localhost', port=port_mapping[rank], stdoutToServer=True, stderrToServer=True)
print(os.getpid())

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Import images and spectra in parallel')
    parser.add_argument('input_path', metavar="input", type=str,
                        help="data folder that includes folders images and spectra")
    parser.add_argument('output_path', metavar="output", type=str,
                        help="path to HDF5 file, does not need to exist")
    parser.add_argument('-t', '--truncate', action='store_const', const=True,
                        help="Should truncate the file if exists?")

    subparsers = parser.add_subparsers(help='commands', dest="command")
    create_parser = subparsers.add_parser("create",
                                          help="This command allows you to create the whole h5 file in one go.")
    update_parser = subparsers.add_parser("update",
                                          help="""These commands allow you to recreate specific parts of the HISS 
                                                  cube h5 file. They all work in the way that there respective group 
                                                  structure within the h5 file gets deleted and recreated from scratch at 
                                                  the moment.""")
    update_parser.add_argument('fits-tables', action='store_true',
                               help="Recreate the FITS paths and serialized headers tables.")
    update_parser.add_argument('semi-sparse-structure', action='store_true',
                               help="Recreate the semi-sparse group and everything beneath.")
    update_parser.add_argument('semi-sparse-data', action='store_true',
                               help="Update all the image and spectra datasets within the semi-sparse group.")
    update_parser.add_argument('image-references', action='store_true',
                               help="Recreate the image references for each spectrum.")
    update_parser.add_argument('dense-cube', action='store_true',
                               help="Recreate the dense cube.")

    args = parser.parse_args()

    dependencies = HiSSCubeProvider(args, args.input_path, args.output_path)
    cli_command_invoker = CLICommandInvoker(args, dependencies)
    cli_command_invoker.execute()
    log_proc_stats()
