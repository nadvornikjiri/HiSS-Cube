from hisscube.processors.metadata_strategy_cube_ml import MLProcessorStrategy


class MLProcessor:
    def __init__(self, metadata_strategy: MLProcessorStrategy):
        self.metadata_strategy = metadata_strategy

    def create_3d_cube(self, h5_connector):
        self.metadata_strategy.create_3d_cube(h5_connector)

    def get_spectrum_3d_cube(self, h5_connector, zoom):
        return self.metadata_strategy.get_spectrum_3d_cube(h5_connector, zoom)

    def get_target_count(self, h5_connector):
        return self.metadata_strategy.get_target_count(h5_connector)
