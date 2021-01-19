from Writer import Writer


class ImageWriter(Writer):
    def ingest_image(self, image_path):
        """
        Method that writes an image to the opened HDF5 file (self.f).
        Parameters
        ----------
        image_path  String

        Returns     HDF5 Dataset (already written to the file)
        -------

        """
        self.write_image_metadata(image_path)
        self.metadata, self.data = self.cube_utils.get_multiple_resolution_image(image_path, self.IMG_MIN_RES)
        img_datasets = self.write_img_datasets()
        return img_datasets

    def create_image_index_tree(self):
        """
        Creates the index tree for an image.
        Returns HDF5 group - the one where the image dataset should be placed.
        -------

        """
        cube_grp = self.require_raw_cube_grp()
        spatial_grps = self.require_image_spatial_grp_structure(cube_grp)
        time_grp = self.require_image_time_grp(spatial_grps[0])
        self.add_hard_links(spatial_grps[1:], time_grp)
        img_spectral_grp = self.require_image_spectral_grp(time_grp)
        res_grps = self.require_res_grps(img_spectral_grp)
        return res_grps
