from hisscube.processors.metadata_strategy_cube_ml import MLProcessorStrategy


class MLProcessor:
    def __init__(self, metadata_strategy: MLProcessorStrategy):
        self.metadata_strategy = metadata_strategy

    def create_3d_cube(self, h5_connector):
        self.metadata_strategy.create_3d_cube(h5_connector)

    def get_spectrum_3d_cube(self, h5_connector, zoom):
        return self.metadata_strategy.get_data(h5_connector, zoom)

    def get_target_count(self, h5_connector):
        return self.metadata_strategy.get_target_count(h5_connector)

    def process_data(self, h5_connector, spectra_index_spec_ids_orig_zoom, target_spatial_indices, offset=None,
                     max_range=None, batch_size=None):
        return self.metadata_strategy.process_data(h5_connector, spectra_index_spec_ids_orig_zoom,
                                                   target_spatial_indices, offset, max_range, batch_size)

    def get_targets(self, serial_connector):
        return self.metadata_strategy.get_targets(serial_connector)

    def get_entry_points(self, h5_connector):
        return self.metadata_strategy.get_entry_points(h5_connector)

    def recreate_datasets(self, serial_connector, dense_grp, target_count):
        return self.metadata_strategy.recreate_datasets(serial_connector, dense_grp, target_count)

    def shrink_datasets(self, final_zoom, serial_connector, final_target_cnt):
        return self.metadata_strategy.shrink_datasets(final_zoom, serial_connector, final_target_cnt)

    def recreate_copy_datasets(self, serial_connector, dense_grp, target_count):
        return self.metadata_strategy.recreate_copy_datasets(serial_connector, dense_grp, target_count)

    def copy_slice(self, h5_connector, old_offset, cnt, new_offset):
        return self.metadata_strategy.copy_slice(h5_connector, old_offset, cnt, new_offset)

    def merge_datasets(self, h5_connector):
        return self.metadata_strategy.merge_datasets(h5_connector)
