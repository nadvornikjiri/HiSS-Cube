def set_nx_entry(grp, h5_connector):
    h5_connector.set_attr(grp, "NX_class", "NXentry")


def set_nx_root(grp, h5_connector):
    h5_connector.set_attr(grp, "NX_class", "NXroot")


def set_nx_default(grp, h5_path, h5_connector):
    h5_connector.set_attr(grp, "default", h5_path)


def set_nx_data(grp, h5_connector):
    h5_connector.set_attr(grp, "NX_class", "NXdata")


def set_nx_interpretation(grp, value, h5_connector):
    h5_connector.set_attr(grp, "interpretation", value)


def set_nx_signal(grp, path, h5_connector):
    h5_connector.set_attr(grp, "signal", path)


def set_nx_axes(grp, axis_list_str, h5_connector):
    h5_connector.set_attr(grp, "axes", axis_list_str)


def add_nexus_navigation_metadata(h5_connector, config):
    root = h5_connector.file
    set_nx_root(root, h5_connector)
    dense_cube_grp = root[config.DENSE_CUBE_NAME]
    set_nx_default(root, dense_cube_grp.name, h5_connector)
    zoom_0_grp = dense_cube_grp["0"]
    set_nx_default(dense_cube_grp, zoom_0_grp.name, h5_connector)
    default_ml_grp = zoom_0_grp["ml_image"]
    set_nx_default(zoom_0_grp, default_ml_grp.name, h5_connector)
    default_cutout_ds = default_ml_grp["cutout_3d_cube_zoom_0"]
    set_nx_default(default_ml_grp, default_cutout_ds.name, h5_connector)
