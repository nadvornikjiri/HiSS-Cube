from hisscube.processors.metadata_strategy_cube_visualization import VisualizationProcessorStrategy


class VisualizationProcessor:

    def __init__(self, metadata_strategy: VisualizationProcessorStrategy):
        self.metadata_strategy = metadata_strategy

    def create_visualization_cube(self, h5_connector):
        self.metadata_strategy.create_visualization_cube(h5_connector)

    def read_spectral_cube_table(self, h5_connector, zoom):
        return self.metadata_strategy.read_spectral_cube_table(h5_connector, zoom)

    def write_VOTable(self, output_path):
        self.metadata_strategy.write_VOTable(output_path)

    def write_FITS(self, output_path):
        self.metadata_strategy.write_VOTable(output_path)
