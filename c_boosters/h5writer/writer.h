//
// Created by caucau on 2/28/22.
//

#ifndef WRITER_WRITER_H
#define WRITER_WRITER_H

#include <Python.h>

void process_h5_dict(PyObject *self, PyObject *args);

bool equals_str(PyObject *key, char *test_str);

hid_t create_h5_file(const char *path);

void process_tree(PyObject *node, hid_t h5_parent_grp, const char *child_grp_name, long res_zoom, hid_t *orig_res_ds);

void write_attrs(hid_t h5_obj, PyObject *attrs, hid_t *orig_res_ds);

void write_attr(PyObject *val, PyObject *key, hid_t h5_obj, hid_t *orig_res_ds);


bool should_track_order(PyObject *pObject);

const char *get_bytes(PyObject *pyObj, PyObject **repr, PyObject **str);

void free_bytes(PyObject *repr, PyObject *str);

PyObject *get_attr_node(PyObject *node);

hid_t create_image_ds(hid_t grp, PyObject *image_node);

void get_image_dims(PyObject *image_node, hsize_t *x, hsize_t *y, hsize_t *z);

const char *get_ds_name(PyObject *node, PyObject **repr, PyObject **str);

hid_t create_spectrum_ds(hid_t grp, PyObject *spectrum_node);

void get_spectrum_dims(PyObject *spectrum_node, hsize_t *x, hsize_t *y);

hid_t create_regref_ds(hid_t grp, PyObject *node);

void get_regref_dims(PyObject *regref_node, hsize_t *x);


long get_res_zoom(PyObject *attr_node);

void process_attributes(hid_t ds, PyObject *value, long res_zoom, hid_t *orig_res_ds);

#endif //WRITER_WRITER_H

