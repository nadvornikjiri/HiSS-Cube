import h5py
import numpy as np
from timeit import default_timer as timer
import csv

FILE_NAME = "sdss.txt"
H5_PATH = "sdss.h5"
CHUNK_SIZE = (128, 128, 2)


def get_property_list(dataset_shape):
    dataset_type = h5py.h5t.py_create(np.dtype('f4'))
    dcpl = h5py.h5p.create(h5py.h5p.DATASET_CREATE)
    dcpl.set_alloc_time(h5py.h5d.ALLOC_TIME_EARLY)
    dcpl.set_fill_time(h5py.h5d.FILL_TIME_NEVER)
    space = h5py.h5s.create_simple(dataset_shape)
    return dcpl, space, dataset_type


with open(FILE_NAME) as f:
    # timing and logging related stuff
    timings_log_csv_file = open("image_timings.csv", "w", newline='')
    timings_logger = csv.writer(timings_log_csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    timings_logger.writerow(["Dataset count", "Time"])
    start = timer()
    check = 100
    dst_cnt = 0

    h5f = h5py.File(H5_PATH, 'w', driver="core", block_size=18000000000)
    lines = f.readlines()
    for line in lines:
        if dst_cnt % check == 0 and dst_cnt / check > 0:  # timing and loggin related stuff
            end = timer()
            print("Dataset cnt: %05d, 100 images done in %.4fs" % (dst_cnt, (end - start)))
            timings_logger.writerow([dst_cnt, (end - start)])
            start = end
        dst_cnt += 1
        if dst_cnt == 1100:
            break
        # line parsing
        path, ds_dims = line.split(" @ ")
        path.replace("\\ ", "")
        img_data_shape = tuple([int(i) for i in ds_dims.strip('{}\n').split(',')])

        # dataset creation
        dcpl, space, img_data_dtype = get_property_list(img_data_shape)
        lcpl = h5py.h5p.create(h5py.h5p.LINK_CREATE)
        lcpl.set_create_intermediate_group(True)
        if CHUNK_SIZE:
            dcpl.set_chunk(CHUNK_SIZE)
        dsid = h5py.h5d.create(h5f.id, path.encode(), img_data_dtype, space, dcpl=dcpl, lcpl=lcpl)

    timings_log_csv_file.close()
    h5f.close()
