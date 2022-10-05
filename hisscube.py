# hisscube.py
import os
import argparse

from hisscube.dependency_injector import HiSSCubeProvider
from hisscube.command import CLICommandInvoker

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Import images and spectra in parallel')
    parser.add_argument('input_path', metavar="input", type=str,
                        help="data folder that includes folders images and spectra")
    parser.add_argument('output_path', metavar="output", type=str,
                        help="path to HDF5 file, does not need to exist")
    parser.add_argument('--truncate', action='store_true', help="Truncate existing Hdf5 file?")
    parser.add_argument('--image-pattern', dest='image_pattern', action='store', nargs='?', type=str,
                        help="Regex pattern to match the images towards.")
    parser.add_argument('--spectra-pattern', dest='spectra_pattern', action='store', nargs='?', type=str,
                        help="Regex pattern to match the spectra towards.")

    subparsers = parser.add_subparsers(help='commands', dest="command")
    create_parser = subparsers.add_parser("create",
                                          help="This command allows you to create the whole h5 file in one go.")
    update_parser = subparsers.add_parser("update",
                                          help="""These commands allow you to recreate specific parts of the HISS 
                                                  cube h5 file. They all work in the way that there respective group 
                                                  structure within the h5 file gets deleted and recreated from scratch at 
                                                  the moment.""")
    update_parser.add_argument('--fits-metadata-cache', action='store_true',
                               help="Recreate the FITS paths and serialized headers tables.")
    update_parser.add_argument('--metadata', action='store_true',
                               help="Recreate the semi-sparse group and everything beneath.")
    update_parser.add_argument('--data', action='store_true',
                               help="Update all the image and spectra datasets within the semi-sparse group.")
    update_parser.add_argument('--link', action='store_true',
                               help="Recreate the image references for each spectrum.")
    update_parser.add_argument('--visualization-cube', action='store_true',
                               help="Recreate the Visualization cube.")
    update_parser.add_argument('--ml-cube', action='store_true',
                               help="Recreate the Machine Learning cube.")

    args = parser.parse_args()

    dependencies = HiSSCubeProvider(args.output_path, input_path=args.input_path, image_pattern=args.image_pattern,
                                    spectra_pattern=args.spectra_pattern)
    cli_command_invoker = CLICommandInvoker(args, dependencies)
    cli_command_invoker.execute()
    dependencies.mpi_helper.log_proc_stats()
