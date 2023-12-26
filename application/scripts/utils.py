# python 3.10 required
import numpy as np
import pandas as pd
import sys
import pycountry
import pickle

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


class CountryPreprocessor:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def process(self):
        self.df['multinoally'] = self.df['multinoally'].fillna(0)
        self.df = self.df.drop(
            ['bilatnoally_mp', 'multinoally_mp', 'CulturalDist_KogutSingh_4', 'CulturalDist_KogutSingh_6',
             'cowc_source_yr', 'cowc_dest_yr'], axis=1)
        self.df = self.df.dropna()

        # scaling
        indicator = ['colonized', 'BIT', 'conflicts_n', 'rta', 'sanctionsinplace', 'cowc_source', 'cowc_dest', 'year']
        columns_to_scale = [col for col in self.df.columns if col not in indicator]
        with open('../../data/model/country-scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)

        self.df[columns_to_scale] = scaler.transform(self.df[columns_to_scale])

        return self.df


def convert_iso3_to_iso2(iso3):
    country = pycountry.countries.get(alpha_3=iso3)
    if country:
        return country.alpha_2
    else:
        return np.nan

def get_country_name(iso2):
    try:
        return pycountry.countries.get(alpha_2=iso2).name
    except AttributeError:
        return "Unknown country code"


class FdiPreprocessor:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    # ISO 3 to ISO 2

    def process(self):
        try:
            self.df['invest_actual'] = self.df['invest_actual'].fillna(0)
            self.df['peers'] = self.df['peers'].fillna(0)
        except AttributeError:
            #if only one observation than this is always Attribute error
            pass

        threshold = 0.4
        max_nan_count = threshold * len(self.df)
        self.df = self.df.dropna(axis=1, thresh=max_nan_count)

        drop_cols = ['bvdid', 'ticker', 'Administrative',
                     'Economic_pooled',
                     'Political_pooled', 'admindistance_KS',
                     'ecodistance_KS', 'peers', 'gdp.percapita',
                     'gdp.annualgrowth', 'population', 'fdi.netinflows',
                     'natural.resource.rents', 'patents', 'polcon3_val',
                     'polcon5_val', 'icrg', 'polity2']
        self.df = self.df.drop(drop_cols, axis=1)

        # drop rows that have NaN
        self.df = self.df.dropna()

        # scaling
        indicator = ['invest_actual', 'cowc3_source',
                     'cowc3_dest', 'year', 'sic', 'foundingyear',
                     'isin']
        columns_to_scale = [col for col in self.df.columns if col not in indicator]
        with open('../../data/model/fdi-scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)

        self.df[columns_to_scale] = scaler.transform(self.df[columns_to_scale])
        self.df['invest_actual'] = self.df['invest_actual'].astype('int64')
        self.df['cowc_source'] = self.df['cowc3_source'].apply(convert_iso3_to_iso2)
        self.df['cowc_dest'] = self.df['cowc3_dest'].apply(convert_iso3_to_iso2)

        self.df = self.df.drop(['cowc3_source', 'cowc3_dest', 'isin'], axis=1)

        return self.df


class Merger:
    def __init__(self, df_country: pd.DataFrame, df_fdi: pd.DataFrame):
        self.df_country = df_country
        self.df_fdi = df_fdi

        #simulate a tupple of company and country
        result = pd.merge(df_fdi, df_country, left_on=['year', 'cowc_source'],
                          right_on=['year', 'cowc_source'], how='inner')
        result.sort_values('year', inplace=True)

        #keep the destination country data
        result['cowc_dest'] = result['cowc_dest_y']
        #result['cowc_dest'] = result['cowc_dest_y']
        result.drop(['cowc_dest_x','cowc_dest_y'], axis=1, inplace=True)
        #print(result.columns)
        with open('../data/model/result-scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)

        print(result.columns)
        print(result.shape)

        columns_to_scale = ['year', 'foundingyear']
        result[columns_to_scale] = scaler.transform(result[columns_to_scale])
        # OneHot Encoding
        columns_to_encode = ['sic', 'cowc_source', 'cowc_dest']
        result = pd.get_dummies(result, columns=columns_to_encode, dtype=int)
        self.result = result.drop("isin", axis=1)

    def get_result(self):
        return self.result
