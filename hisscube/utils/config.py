import configparser
import pathlib


class Config:
    def __init__(self):
        lib_path = pathlib.Path(__file__).parent.absolute()
        self.config = configparser.ConfigParser(allow_no_value=True)
        self.config.read("%s/../config.ini" % lib_path)

        self.CREATE_REFERENCES = self.config.getboolean('Builder', 'CREATE_REFERENCES')
        self.CREATE_VISUALIZATION_CUBE = self.config.getboolean('Builder', 'CREATE_VISUALIZATION_CUBE')
        self.CREATE_ML_CUBE = self.config.getboolean('Builder', 'CREATE_ML_CUBE')
        self.MPIO = self.config.getboolean('Builder', 'MPIO')
        self.PARALLEL_MODE = self.config.get('Builder', 'PARALLEL_MODE')
        self.C_BOOSTER = self.config.getboolean('Builder', 'C_BOOSTER')
        self.METADATA_STRATEGY = self.config.get('Builder', 'METADATA_STRATEGY')
        self.DATASET_STRATEGY_CHUNKED = self.config.getboolean('Builder', 'DATASET_STRATEGY_CHUNKED')

        self.IMAGE_CUTOUT_SIZE = self.config.getint('Handler', 'IMAGE_CUTOUT_SIZE')
        self.IMG_ZOOM_CNT = self.config.getint('Handler', 'IMG_ZOOM_CNT')
        self.SPEC_ZOOM_CNT = self.config.getint('Handler', 'SPEC_ZOOM_CNT')
        self.IMG_SPAT_INDEX_ORDER = self.config.getint('Handler', 'IMG_SPAT_INDEX_ORDER')
        self.IMG_DIAMETER_ANG_MIN = self.config.getfloat('Handler', 'IMG_DIAMETER_ANG_MIN')
        self.SPEC_SPAT_INDEX_ORDER = self.config.getint('Handler', 'SPEC_SPAT_INDEX_ORDER')
        self.CHUNK_SIZE = self.config.get('Handler', 'CHUNK_SIZE')
        self.ORIG_CUBE_NAME = self.config.get('Handler', 'ORIG_CUBE_NAME')
        self.DENSE_CUBE_NAME = self.config.get('Handler', 'DENSE_CUBE_NAME')
        self.INCLUDE_ADDITIONAL_METADATA = self.config.getboolean('Handler', 'INCLUDE_ADDITIONAL_METADATA')
        self.INIT_ARRAY_SIZE = self.config.getint('Handler', 'INIT_ARRAY_SIZE')
        self.FITS_MEM_MAP = self.config.getboolean('Handler', 'FITS_MEM_MAP')
        self.LOG_LEVEL = self.config.get('Handler', 'LOG_LEVEL')

        self.COMPRESSION = self.config.get('Writer', 'COMPRESSION')
        self.COMPRESSION_OPTS = self.config.get('Writer', 'COMPRESSION_OPTS')
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
        self.FITS_HEADER_BUF_SIZE = self.config.getint('Writer', 'FITS_HEADER_BUF_SIZE')
        self.FITS_MAX_PATH_SIZE = self.config.getint('Writer', 'FITS_MAX_PATH_SIZE')
        self.BATCH_SIZE = self.config.getint('Writer', 'BATCH_SIZE')
        self.POLL_INTERVAL = self.config.getfloat('Writer', 'POLL_INTERVAL')

        self.OUTPUT_HEAL_ORDER = self.config.getint('Reader', 'OUTPUT_HEAL_ORDER')

        self.APPLY_TRANSMISSION_ONLINE = self.config.get('Processor', 'APPLY_TRANSMISSION_ONLINE')

        self.REBIN_MIN = self.config.getfloat('SDSS', 'REBIN_MIN')
        self.REBIN_MAX = self.config.getfloat('SDSS', 'REBIN_MAX')
        self.REBIN_SAMPLES = self.config.getint('SDSS', 'REBIN_SAMPLES')
        self.APPLY_REBIN = self.config.getboolean('SDSS', 'APPLY_REBIN')
        self.APPLY_TRANSMISSION_CURVE = self.config.getboolean('SDSS', 'APPLY_TRANSMISSION_CURVE')
        self.IMG_RES_X = self.config.getint('SDSS', 'IMG_RES_X')
        self.IMG_RES_Y = self.config.getint('SDSS', 'IMG_RES_Y')

