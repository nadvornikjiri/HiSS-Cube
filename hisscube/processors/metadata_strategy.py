from abc import ABC, abstractmethod

import healpy

from hisscube.utils.astrometry import get_image_lower_res_wcs
from hisscube.utils.config import Config
from hisscube.utils.io import H5Connector


class MetadataStrategy(ABC):
    def __init__(self, config: Config):
        self.config = config

    @abstractmethod
    def require_spatial_grp(self, h5_connector, order, prev, coord):
        raise NotImplementedError

    @abstractmethod
    def add_metadata(self, h5_connector, metadata, datasets):
        raise NotImplementedError


class TreeStrategy(MetadataStrategy):

    def require_spatial_grp(self, h5_connector, order, prev, coord):
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

    def add_metadata(self, h5_connector, metadata, datasets):
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
        image_fits_header = dict(metadata)
        for res_idx, ds in enumerate(datasets):
            if res_idx > 0:
                h5_connector.set_attr_ref(ds, "orig_res_link", datasets[0])
                orig_image_fits_header = h5_connector.read_serialized_fits_header(datasets[0])
                if h5_connector.get_attr(ds, "mime-type") == "image":
                    image_fits_header = get_image_lower_res_wcs(orig_image_fits_header, image_fits_header, res_idx)
            naxis = len(h5_connector.get_shape(ds))
            image_fits_header["NAXIS"] = naxis
            for axis in range(naxis):
                image_fits_header["NAXIS%d" % (axis)] = h5_connector.get_shape(ds)[axis]
            h5_connector.write_serialized_fits_header(ds, image_fits_header)


class DatasetStrategy(MetadataStrategy):

    def add_metadata(self, h5_connector, metadata, datasets):
        pass

    def require_spatial_grp(self, h5_connector, order, prev, coord):
        pass
