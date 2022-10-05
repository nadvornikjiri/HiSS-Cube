import healpy

from hisscube.processors.metadata_strategy import MetadataStrategy, write_naxis_values, get_lower_res_image_metadata
from hisscube.utils.astrometry import get_optimized_wcs, get_cutout_bounds
from hisscube.utils.io import get_time_from_image
from hisscube.utils.io_strategy import get_orig_header


def require_spatial_grp(h5_connector, order, prev, coord):
    """
    Returns the HEALPix group structure.
    Parameters
    ----------
    order   int
    prev    HDF5 group
    coord   (float, float)

    Returns
    -------

    """
    nside = 2 ** order
    healID = healpy.ang2pix(nside, coord[0], coord[1], lonlat=True, nest=True)
    grp = h5_connector.require_group(prev, str(healID))  # TODO optimize to 8-byte string?
    h5_connector.set_attr(grp, "type", "spatial")
    return grp


class TreeStrategy(MetadataStrategy):

    def add_metadata(self, h5_connector, metadata, datasets, img_cnt=None, fits_name=None):
        """
        Adds metadata to the HDF5 data sets of the same image or spectrum in multiple resolutions. It also modifies the
        metadata for image where needed and adds the COMMENT and HISTORY attributes as datasets for optimization
        purposes.
        Parameters
        ----------
        datasets    [HDF5 Datasets]

        Returns
        -------

        """
        fits_header = dict(metadata)
        for res_idx, ds in enumerate(datasets):
            if res_idx > 0:
                fits_header = self._get_lower_res_metadata(datasets, ds, h5_connector, fits_header, res_idx)
            ds_shape = h5_connector.get_shape(ds)
            write_naxis_values(fits_header, ds_shape)
            h5_connector.write_serialized_fits_header(ds, fits_header)

    @staticmethod
    def get_cutout_bounds_from_spectrum(h5_connector, image_ds, spectrum_ds, image_cutout_size, res_idx):
        orig_image_header = get_orig_header(h5_connector, image_ds)
        orig_spectrum_header = get_orig_header(h5_connector, spectrum_ds)
        time = get_time_from_image(orig_image_header)
        wl = int(image_ds.name.split('/')[-3])
        image_fits_header = h5_connector.read_serialized_fits_header(image_ds)
        w = get_optimized_wcs(image_fits_header)
        cutout_bounds = get_cutout_bounds(image_fits_header, res_idx, orig_spectrum_header,
                                          image_cutout_size)
        return cutout_bounds, time, w, wl

    @staticmethod
    def _get_lower_res_metadata(datasets, ds, h5_connector, fits_header, res_idx):
        if h5_connector.get_attr(ds, "mime-type") == "image":
            h5_connector.set_attr_ref(ds, "orig_res_link", datasets[0])
            orig_image_fits_header = h5_connector.read_serialized_fits_header(datasets[0])
            fits_header = get_lower_res_image_metadata(fits_header, orig_image_fits_header, res_idx)
        return fits_header
