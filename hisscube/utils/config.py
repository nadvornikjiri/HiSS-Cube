import configparser
import pathlib


class Config:
    def __init__(self):

        lib_path = pathlib.Path(__file__).parent.absolute()
        self.config = configparser.ConfigParser(allow_no_value=True, inline_comment_prefixes='#')
        self.config.read("%s/../config.ini" % lib_path)

        self.CREATE_REFERENCES = self.config.getboolean('Builder', 'CREATE_REFERENCES')
        self.CREATE_VISUALIZATION_CUBE = self.config.getboolean('Builder', 'CREATE_VISUALIZATION_CUBE')
        self.CREATE_ML_CUBE = self.config.getboolean('Builder', 'CREATE_ML_CUBE')
        self.MPIO = self.config.getboolean('Builder', 'MPIO')
        self.PARALLEL_MODE = self.config.get('Builder', 'PARALLEL_MODE')
        self.C_BOOSTER = self.config.getboolean('Builder', 'C_BOOSTER')
        self.METADATA_STRATEGY = self.config.get('Builder', 'METADATA_STRATEGY')
        self.DATASET_STRATEGY_CHUNKED = self.config.getboolean('Builder', 'DATASET_STRATEGY_CHUNKED')
        self.PATH_WAIT_TOTAL = self.config.getboolean('Builder', 'PATH_WAIT_TOTAL')

        self.IMAGE_CUTOUT_SIZE = self.config.getint('Handler', 'IMAGE_CUTOUT_SIZE')
        self.IMG_ZOOM_CNT = self.config.getint('Handler', 'IMG_ZOOM_CNT')
        self.SPEC_ZOOM_CNT = self.config.getint('Handler', 'SPEC_ZOOM_CNT')
        self.IMG_SPAT_INDEX_ORDER = self.config.getint('Handler', 'IMG_SPAT_INDEX_ORDER')
        self.SPEC_SPAT_INDEX_ORDER = self.config.getint('Handler', 'SPEC_SPAT_INDEX_ORDER')
        self.IMAGE_CHUNK_SIZE = self.config.get('Handler', 'IMAGE_CHUNK_SIZE')
        self.ML_CUBE_CHUNK_SIZE = self.config.getint('Handler', 'ML_CUBE_CHUNK_SIZE')
        self.SPARSE_CUBE_NAME = self.config.get('Handler', 'SPARSE_CUBE_NAME')
        self.DENSE_CUBE_NAME = self.config.get('Handler', 'DENSE_CUBE_NAME')
        self.INCLUDE_ADDITIONAL_METADATA = self.config.getboolean('Handler', 'INCLUDE_ADDITIONAL_METADATA')
        self.INIT_ARRAY_SIZE = self.config.getint('Handler', 'INIT_ARRAY_SIZE')
        self.FITS_MEM_MAP = self.config.getboolean('Handler', 'FITS_MEM_MAP')
        self.LOG_LEVEL = self.config.get('Handler', 'LOG_LEVEL')
        self.METADATA_CHUNK_SIZE = self.config.getint('Handler', 'METADATA_CHUNK_SIZE')

        self.COMPRESSION = self.config.get('Writer', 'COMPRESSION')
        try:
            self.COMPRESSION_OPTS = self.config.getint('Writer', 'COMPRESSION_OPTS')
        except TypeError:
            self.COMPRESSION_OPTS = None
        self.FLOAT_COMPRESS = self.config.getboolean('Writer', 'FLOAT_COMPRESS')
        self.SHUFFLE = self.config.getboolean('Writer', 'SHUFFLE')
        self.IMAGE_PATTERN = self.config.get('Writer', 'IMAGE_PATTERN')
        self.SPECTRA_PATTERN = self.config.get('Writer', 'SPECTRA_PATTERN')
        self.MAX_CUTOUT_REFS = self.config.getint('Writer', 'MAX_CUTOUT_REFS')
        self.LIMIT_IMAGE_COUNT = self.config.getint('Writer', 'LIMIT_IMAGE_COUNT')
        self.LIMIT_SPECTRA_COUNT = self.config.getint('Writer', 'LIMIT_SPECTRA_COUNT')
        self.FITS_IMAGE_MAX_HEADER_SIZE = self.config.getint('Writer', 'FITS_IMAGE_MAX_HEADER_SIZE')
        self.FITS_SPECTRUM_MAX_HEADER_SIZE = self.config.getint('Writer', 'FITS_SPECTRUM_MAX_HEADER_SIZE')
        self.MAX_STORED_IMAGE_HEADERS = self.config.getint('Writer', 'MAX_STORED_IMAGE_HEADERS')
        self.MAX_STORED_SPECTRA_HEADERS = self.config.getint('Writer', 'MAX_STORED_SPECTRA_HEADERS')
        self.FITS_MAX_PATH_SIZE = self.config.getint('Writer', 'FITS_MAX_PATH_SIZE')
        self.MAX_DS_PATH_SIZE = self.config.getint('Writer', 'MAX_DS_PATH_SIZE')

        self.FITS_HEADER_BATCH_SIZE = self.config.getint('MPI', 'FITS_HEADER_BATCH_SIZE')
        self.METADATA_BATCH_SIZE = self.config.getint('MPI', 'METADATA_BATCH_SIZE')
        self.IMAGE_DATA_BATCH_SIZE = self.config.getint('MPI', 'IMAGE_DATA_BATCH_SIZE')
        self.SPECTRUM_DATA_BATCH_SIZE = self.config.getint('MPI', 'SPECTRUM_DATA_BATCH_SIZE')
        self.POLL_INTERVAL = self.config.getfloat('MPI', 'POLL_INTERVAL')
        self.LINK_BATCH_SIZE = self.config.getint('MPI', 'LINK_BATCH_SIZE')
        self.ML_BATCH_SIZE = self.config.getint('MPI', 'ML_BATCH_SIZE')
        self.CACHE_INDEX_FOR_LINKING = self.config.getboolean('MPI', 'CACHE_INDEX_FOR_LINKING')
        self.CACHE_WCS_FOR_LINKING = self.config.getboolean('MPI', 'CACHE_WCS_FOR_LINKING')

        self.OUTPUT_HEAL_ORDER = self.config.getint('Reader', 'OUTPUT_HEAL_ORDER')

        self.APPLY_TRANSMISSION_ONLINE = self.config.get('Processor', 'APPLY_TRANSMISSION_ONLINE')

        self.REBIN_MIN = self.config.getfloat('SDSS', 'REBIN_MIN')
        self.REBIN_MAX = self.config.getfloat('SDSS', 'REBIN_MAX')
        self.REBIN_SAMPLES = self.config.getint('SDSS', 'REBIN_SAMPLES')
        self.APPLY_REBIN = self.config.getboolean('SDSS', 'APPLY_REBIN')
        self.APPLY_TRANSMISSION_CURVE = self.config.getboolean('SDSS', 'APPLY_TRANSMISSION_CURVE')
        self.IMG_RES_X = self.config.getint('SDSS', 'IMG_RES_X')
        self.IMG_RES_Y = self.config.getint('SDSS', 'IMG_RES_Y')
        self.IMG_DIAMETER_ANG_MIN = self.config.getfloat('SDSS', 'IMG_DIAMETER_ANG_MIN')
        self.FILTER_CNT = self.config.getint('SDSS', 'FILTER_CNT')
        self.SPECTRUM_FIBER_DIAMETER = self.config.getfloat('SDSS', 'SPECTRUM_FIBER_DIAMETER')
        self.IMAGE_PIXEL_SIZE = self.config.getfloat('SDSS', 'IMAGE_PIXEL_SIZE')

        try:
            self.IMG_X_SIZE_ANG_MIN = self.config.getfloat('SDSS', 'IMG_X_SIZE_ANG_MIN')
        except ValueError:
            self.IMG_X_SIZE_ANG_MIN = None
        try:
            self.IMG_Y_SIZE_ANG_MIN = self.config.getfloat('SDSS', 'IMG_Y_SIZE_ANG_MIN')
        except ValueError:
            self.IMG_Y_SIZE_ANG_MIN = None
