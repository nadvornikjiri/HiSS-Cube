import pandas as pd
import ujson
from astropy.table import Table

from hisscube.utils.config import Config
from hisscube.utils.io import get_spectrum_header_dataset


def convert_str_cols(df):
    str_cols = df.select_dtypes([object])
    for str_col in str_cols:
        df[str_col] = df[str_col].astype('|S')


class SFRProcessor:

    @staticmethod
    def write_sfr(pytables_connector, gal_info_path, gal_sfr_path):
        ignore_info_cols = ['PHOTOID', 'PLUG_MAG', 'SPECTRO_MAG', 'KCOR_MAG', 'KCOR_MODEL_MAG']

        data_info = Table.read(gal_info_path, format='fits')
        for col in ignore_info_cols:
            del data_info[col]
        data_info.convert_bytestring_to_unicode()
        gal_info_df = data_info.to_pandas()
        convert_str_cols(gal_info_df)
        data_fibsfr = Table.read(gal_sfr_path, format='fits')
        sfr_df = data_fibsfr.to_pandas()
        convert_str_cols(sfr_df)
        df_concat_sfr = pd.concat([gal_info_df, sfr_df], axis=1)
        pytables_connector.file.put("star_formation_rates", df_concat_sfr)
        return df_concat_sfr

    @staticmethod
    def get_spectrum_metadata(h5py_connector):
        spectrum_original_headers_data = get_spectrum_header_dataset(h5py_connector)[:]["header"]
        parsed_headers = [ujson.decode(header) for header in spectrum_original_headers_data]
        parsed_headers_df = pd.DataFrame.from_dict(parsed_headers)
        convert_str_cols(parsed_headers_df)
        return parsed_headers_df

    @staticmethod
    def write_spec_metadata_with_sfr(pytables_connector, parsed_spectrum_headers, sfr_table):
        headers_sfr_merged_df = pd.merge(parsed_spectrum_headers, sfr_table, on=["PLATEID", "MJD", "FIBERID"],
                                         how="left")
        convert_str_cols(headers_sfr_merged_df)
        pytables_connector.file.put("fits_spectra_metadata_star_formation_rates", headers_sfr_merged_df)
        return headers_sfr_merged_df
