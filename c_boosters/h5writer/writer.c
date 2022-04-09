//
// Created by caucau on 2/28/22.
//
//#define DEBUG 1

#define LOG_DS_CHUNK 100
#define LOG_GRP_CHUNK 100

#include <stdio.h>
#include <Python.h>
#include "hdf5.h"
#include "writer.h"

long ds_cnt, grp_cnt;
clock_t start, end;
double elapsed_time;
FILE *logfp;

int open_grp_cnt = 0, open_ds_cnt = 0;

void log_timing();

void write_to_csv();

bool contains_str(PyObject *key, char *test_string);

void process_h5_dict(PyObject *self, PyObject *args) {

    PyObject *pyDict;
    char *h5_path;
    char *timing_log_path;
    hid_t orig_res_ds = 0;
    hid_t parent_grp, child_grp;
    ds_cnt = 0;
    grp_cnt = 0;

    if (!PyArg_ParseTuple(args, "O!ss", &PyDict_Type, &pyDict, &h5_path, &timing_log_path)) {
        return;
    }

    printf("Creating the csv logging file: %s\n", timing_log_path);
    logfp = fopen(timing_log_path, "w+");

    fprintf(logfp, "Dataset count,Group count,Time\n");

    printf("Writing HDF5 file %s.\n", h5_path);
    hid_t h5_file = create_h5_file(h5_path);
    start = clock();
    PyObject *key, *value;
    Py_ssize_t pos = 0;
    while (PyDict_Next(pyDict, &pos, &key, &value)) {
        if (equals_str(key, "semi_sparse_cube")) {
            parent_grp = H5Gcreate(h5_file, "semi_sparse_cube", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            open_grp_cnt ++;
            grp_cnt ++;
            if (parent_grp < 0) {
                H5Eprint(H5E_DEFAULT, stderr);
            }
            child_grp = process_tree(value, parent_grp, NULL, -1, &orig_res_ds);
            H5Gclose(child_grp);
            open_grp_cnt --;
            H5Dclose(orig_res_ds);  //closing last original resolution ds kept in memory
            open_ds_cnt --;
#ifdef DEBUG
            printf("Grp cnt: %d\n", open_grp_cnt);
            printf("DS cnt: %d\n", open_ds_cnt);
#endif
        } else if (equals_str(key, "attrs")) {
            write_attrs(h5_file, value, 0);
        }
    }
    H5Fflush(h5_file, H5F_SCOPE_GLOBAL);
    H5Fclose(h5_file);
    fclose(logfp);
}

hid_t create_h5_file(const char *path) {
    hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_sec2(fapl); //just to be sure
    H5Pset_libver_bounds(fapl, H5F_LIBVER_LATEST, H5F_LIBVER_LATEST);
    hid_t fcpl = H5Pcreate(H5P_FILE_CREATE);
    H5Pset_file_space_strategy(fcpl, H5F_FSPACE_STRATEGY_PAGE, 0, 1);
    hid_t hfile = H5Fcreate(path, H5F_ACC_TRUNC, fcpl, fapl);
    if (hfile == H5I_INVALID_HID) {
        H5Eprint(H5E_DEFAULT, stderr);
        exit(1);
    }
    return hfile;
}

hid_t process_tree(PyObject *node, hid_t parent_grp, const char *child_grp_name, long res_zoom, hid_t *orig_res_ds) {
    hid_t ds;
    hid_t child_grp;
    if (child_grp_name != NULL) {
        hid_t gcpl = H5Pcreate(H5P_GROUP_CREATE);
        if (should_track_order(node)) {

            H5Pset_link_creation_order(gcpl, H5P_CRT_ORDER_TRACKED | H5P_CRT_ORDER_INDEXED);
            H5Pset_attr_creation_order(gcpl, H5P_CRT_ORDER_TRACKED | H5P_CRT_ORDER_INDEXED);
        }
#ifdef DEBUG
        {
            size_t len = H5Iget_name(parent_grp, NULL, 0) + 1;
            char *buffer;
            buffer = (char *) malloc(len * sizeof(char));
            H5Iget_name(parent_grp, buffer, len);
            printf("Created group %s in parent group %s\n", child_grp_name, buffer);
            free(buffer);
        }
#endif

        child_grp = H5Gcreate(parent_grp, child_grp_name, H5P_DEFAULT, gcpl, H5P_DEFAULT);
        open_grp_cnt ++;
        grp_cnt ++;
        log_timing();
        if (child_grp < 0) {
            H5Eprint(H5E_DEFAULT, stderr);
        }
        H5Pclose(gcpl);
        PyObject *attr_node = get_attr_node(node);
        write_attrs(child_grp, attr_node, 0);
        parent_grp = child_grp;
    }
    PyObject *key, *value;
    Py_ssize_t pos = 0;
    while (PyDict_Next(node, &pos, &key, &value)) {
        if (!equals_str(key, "name") &&
            !equals_str(key, "attrs") &&
            !equals_str(key, "track_order")) {  //these are handled beneath ds or grp creation itself
            if (equals_str(key, "image_dataset")) {
                ds = create_image_ds(parent_grp, value);
                process_attributes(ds, value, res_zoom, orig_res_ds);
                log_timing();
            } else if (equals_str(key, "spectrum_dataset")) {
                ds = create_spectrum_ds(parent_grp, value);
                process_attributes(ds, value, res_zoom, orig_res_ds);
                log_timing();
            } else if (contains_str(key, "image_cutouts")) { //TODO check for dataset dtype, now it's hard-coded
                ds = create_regref_ds(parent_grp, value);
                H5Dclose(ds);
                open_ds_cnt --;
                log_timing();
            } else {
                PyObject *grp_repr, *grp_str;
                child_grp_name = get_bytes(key, &grp_repr, &grp_str);
                PyObject *attr_node = get_attr_node(value);
                long res_zoom = get_res_zoom(attr_node);
                child_grp = process_tree(value, parent_grp, child_grp_name, res_zoom, orig_res_ds);
                H5Gclose(child_grp);
                open_grp_cnt --;
                free_bytes(grp_repr, grp_str);
            }
        }
    }
    return parent_grp;
}

void log_timing() {
    if (ds_cnt % LOG_DS_CHUNK == 0) {
        write_to_csv();
    } else if (ds_cnt == 0 && (grp_cnt % LOG_GRP_CHUNK == 0)) { //if there are not datasets to write, we log based on grp
        write_to_csv();
    }
}

void write_to_csv() {
    end = clock();
    elapsed_time = ((double) (end - start)) / CLOCKS_PER_SEC;
    fprintf(logfp, "%zu, %zu, %f\n", ds_cnt, grp_cnt, elapsed_time);
    start = end;
}

void process_attributes(hid_t ds, PyObject *value, long res_zoom, hid_t *orig_res_ds) {
    if (res_zoom == 0) {
        if (*orig_res_ds != 0) {  //first image, nothing to close
            H5Dclose(*orig_res_ds); //close the previous original dataset link
            open_ds_cnt --;
        }
        *orig_res_ds = ds;
    }
    PyObject *attr_node = get_attr_node(value);
    write_attrs(ds, attr_node, orig_res_ds);
    if (res_zoom > 0) { //higher zooms don't need to be kept in memory
        H5Dclose(ds);
        open_ds_cnt --;
    }
}


void write_attrs(hid_t h5_obj, PyObject *attrs, hid_t *orig_res_ds) {
    PyObject *key, *value;
    Py_ssize_t pos = 0;
    while (PyDict_Next(attrs, &pos, &key, &value)) {
        write_attr(value, key, h5_obj, orig_res_ds);
    }
}

void write_attr(PyObject *val, PyObject *key, hid_t h5_obj, hid_t *orig_res_ds) {
    hid_t attr, acpl, atype, aspace;
    PyObject *repr, *str;
    const char *key_bytes = get_bytes(key, &repr, &str);
    const char *dtype = val->ob_type->tp_name;
    aspace = H5Screate(H5S_SCALAR);
    if (aspace < 0) {
        H5Eprint(H5E_DEFAULT, stderr);
    }
    if (strcmp(dtype, "int") == 0) {
        long attr_val = PyLong_AsLong(val);
        attr = H5Acreate(h5_obj, key_bytes, H5T_NATIVE_LONG, aspace, H5P_DEFAULT, H5P_DEFAULT);
        if (attr < 0) {
            H5Eprint(H5E_DEFAULT, stderr);
        }
        H5Awrite(attr, H5T_NATIVE_LONG, &attr_val);
        H5Aclose(attr);
    } else if (equals_str(key, "orig_res_link")) {
        hobj_ref_t *ds_ref;
        ds_ref = (hobj_ref_t *) malloc(sizeof(hobj_ref_t) * 1);  //just one dataset reference please
        int status = H5Rcreate(ds_ref, *orig_res_ds, ".", H5R_OBJECT, -1);
        if (status < 0) {
            H5Eprint(H5E_DEFAULT, stderr);
        }
        attr = H5Acreate(h5_obj, key_bytes, H5T_STD_REF_OBJ, aspace, H5P_DEFAULT, H5P_DEFAULT);
        if (attr < 0) {
            H5Eprint(H5E_DEFAULT, stderr);
        }
        status = H5Awrite(attr, H5T_STD_REF_OBJ, ds_ref);
        if (status < 0) {
            H5Eprint(H5E_DEFAULT, stderr);
        }
        H5Aclose(attr);
        free(ds_ref);

    } else if (strcmp(dtype, "str") == 0) {
        PyObject *val_repr, *val_str;
        const char *val_bytes = get_bytes(val, &val_repr, &val_str);
        acpl = H5Pcreate(H5P_ATTRIBUTE_CREATE);
        atype = H5Tcopy (H5T_C_S1);
        H5Tset_size (atype, H5T_VARIABLE);
        H5Tset_cset(atype, H5T_CSET_UTF8);
        attr = H5Acreate(h5_obj, key_bytes, atype, aspace, H5P_DEFAULT, H5P_DEFAULT);
        if (attr < 0) {
            H5Eprint(H5E_DEFAULT, stderr);
        }
        H5Awrite(attr, atype, &val_bytes);
        free_bytes(val_repr, val_str);
        H5Pclose(acpl);
        H5Tclose(atype);
        H5Aclose(attr);
    }
    H5Sclose(aspace);
    free_bytes(repr, str);
}

hid_t create_image_ds(hid_t grp, PyObject *image_node) {
    PyObject *repr, *str;
    hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
    H5Pset_alloc_time(dcpl, H5D_ALLOC_TIME_EARLY);
    H5Pset_fill_time(dcpl, H5D_FILL_TIME_NEVER);
    H5Pset_chunk(dcpl, 3, (hsize_t[]) {128, 128, 2});   //TODO pass this as parameter
    hid_t fspace;
    hsize_t x = 0, y = 0, z = 0;
    get_image_dims(image_node, &x, &y, &z);
    fspace = H5Screate_simple(3, (hsize_t[]) {x, y, z}, NULL);
    hid_t image_ds = H5Dcreate(grp, get_ds_name(image_node, &repr, &str), H5T_NATIVE_FLOAT, fspace,
                               H5P_DEFAULT, dcpl, H5P_DEFAULT);
    open_ds_cnt ++; ds_cnt ++;
    free_bytes(repr, str);\

    if (image_ds < 0) {
        H5Eprint(H5E_DEFAULT, stderr);
    }
#ifdef DEBUG
    size_t len = H5Iget_name(grp, NULL, 0) + 1;
    char *buffer;
    buffer = (char *) malloc(len * sizeof(char));
    H5Iget_name(grp, buffer, len);
    printf("Created image dataset: %s in group %s\n", get_ds_name(image_node, &repr, &str), buffer);
    free_bytes(repr, str);
    free(buffer);
#endif
    H5Sclose(fspace);
    H5Pclose(dcpl);
    return image_ds;
}

hid_t create_spectrum_ds(hid_t grp, PyObject *spectrum_node) {
    PyObject *repr, *str;
    hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
    H5Pset_alloc_time(dcpl, H5D_ALLOC_TIME_EARLY);
    H5Pset_fill_time(dcpl, H5D_FILL_TIME_NEVER);
    hid_t fspace;
    hsize_t x = 0, y = 0;
    get_spectrum_dims(spectrum_node, &x, &y);
    fspace = H5Screate_simple(2, (hsize_t[]) {x, y}, NULL);
    hid_t ds = H5Dcreate(grp, get_ds_name(spectrum_node, &repr, &str), H5T_NATIVE_FLOAT, fspace,
                         H5P_DEFAULT, dcpl, H5P_DEFAULT);
    open_ds_cnt ++; ds_cnt ++;
    free_bytes(repr, str);
    if (ds < 0) {
        H5Eprint(H5E_DEFAULT, stderr);
    }
#ifdef DEBUG
    printf("Created spectral dataset: %s\n", get_ds_name(spectrum_node, &repr, &str));
    free_bytes(repr, str);
#endif

    H5Sclose(fspace);
    H5Pclose(dcpl);
    return ds;
}

hid_t create_regref_ds(hid_t grp, PyObject *node) {
    PyObject *repr, *str;
    hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
    H5Pset_alloc_time(dcpl, H5D_ALLOC_TIME_EARLY);
    H5Pset_fill_time(dcpl, H5D_FILL_TIME_NEVER);
    hid_t fspace;
    hsize_t x = 0;
    get_regref_dims(node, &x);
    fspace = H5Screate_simple(1, (hsize_t[]) {x}, NULL);
    hid_t regref_ds = H5Dcreate(grp, get_ds_name(node, &repr, &str), H5T_STD_REF_DSETREG, fspace,
                                H5P_DEFAULT, dcpl, H5P_DEFAULT);
    open_ds_cnt ++; ds_cnt ++;
    free_bytes(repr, str);
    if (regref_ds < 0) {
        H5Eprint(H5E_DEFAULT, stderr);
    }
#ifdef DEBUG
    printf("Created regref dataset: %s\n", get_ds_name(node, &repr, &str));
    free_bytes(repr, str);
#endif
    H5Sclose(fspace);
    H5Pclose(dcpl);
    return regref_ds;
}

void get_image_dims(PyObject *image_node, hsize_t *x, hsize_t *y, hsize_t *z) {
    PyObject *key, *value;
    Py_ssize_t pos = 0;
    while (PyDict_Next(image_node, &pos, &key, &value)) {
        if (equals_str(key, "shape")) {
            PyArg_ParseTuple(value, "iii", x, y, z);
            return;
        }
    }
}

void get_spectrum_dims(PyObject *spectrum_node, hsize_t *x, hsize_t *y) {
    PyObject *key, *value;
    Py_ssize_t pos = 0;
    while (PyDict_Next(spectrum_node, &pos, &key, &value)) {
        if (equals_str(key, "shape")) {
            PyArg_ParseTuple(value, "ii", x, y);
            return;
        }
    }
}

void get_regref_dims(PyObject *regref_node, hsize_t *x) {
    PyObject *key, *value;
    Py_ssize_t pos = 0;
    while (PyDict_Next(regref_node, &pos, &key, &value)) {
        if (equals_str(key, "shape")) {
            PyArg_ParseTuple(value, "i", x);
            return;
        }
    }
}

const char *get_ds_name(PyObject *node, PyObject **repr, PyObject **str) {
    PyObject *key, *value;
    Py_ssize_t pos = 0;
    while (PyDict_Next(node, &pos, &key, &value)) {
        if (equals_str(key, "name")) {
            return get_bytes(value, repr, str);
        }
    }
    return NULL;
}


PyObject *get_attr_node(PyObject *node) {
    PyObject *key, *value;
    Py_ssize_t pos = 0;
    while (PyDict_Next(node, &pos, &key, &value)) {
        if (equals_str(key, "attrs")) {
            return value;
        }
    }
    return NULL;
}


bool should_track_order(PyObject *node) {
    PyObject *key, *value;
    Py_ssize_t pos = 0;
    while (PyDict_Next(node, &pos, &key, &value)) {
        if (equals_str(key, "track_order")) {
            return 1;
        }
    }
    return 0;
}

long get_res_zoom(PyObject *attr_node) {
    PyObject *key, *value;
    Py_ssize_t pos = 0;
    while (PyDict_Next(attr_node, &pos, &key, &value)) {
        if (equals_str(key, "res_zoom")) {
            return PyLong_AsLong(value);
        }
    }
    return -1;
}


bool equals_str(PyObject *key, char *test_str) {
    PyObject *repr, *str;
    if (key == NULL) {
        return 0;
    }
    const char *bytes = get_bytes(key, &repr, &str);
    int res = (!strcmp(bytes, test_str));
    free_bytes(repr, str);
    return res;
}

bool contains_str(PyObject *key, char *test_str) {
    PyObject *repr, *str;
    if (key == NULL) {
        return 0;
    }
    const char *bytes = get_bytes(key, &repr, &str);
    int res = strstr(bytes, test_str);
    free_bytes(repr, str);
    return res;
}

const char *get_bytes(PyObject *pyObj, PyObject **repr, PyObject **str) {
    *repr = PyObject_Str(pyObj);
    *str = PyUnicode_AsEncodedString(*repr, "utf-8", "~E~");
    const char *bytes = PyBytes_AS_STRING(*str);

    return bytes;
}

void free_bytes(PyObject *repr, PyObject *str) {
    Py_XDECREF(repr);
    Py_DECREF(str);
}

