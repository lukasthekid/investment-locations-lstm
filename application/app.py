import streamlit as st
import pandas as pd
import sys
import os

# sys.path.insert(0, "../scripts")

from scripts.utils import Merger, FdiPreprocessor, CountryPreprocessor
from tensorflow import keras


def get_recommendation(isin, year):
    print(os.getcwd())
    df_fdi = pd.read_csv("../data/preprocessed/preprocessed_fdi.csv")
    df_country = pd.read_csv("../data/preprocessed/preprocessed_country.csv")
    # get the company
    #print(df_fdi[(df_fdi['year'] == year) & (df_fdi['isin'] == isin)].head())
    df_fdi_filtered = df_fdi[(df_fdi['year'] == year) & (df_fdi['isin'] == isin)].iloc[0:1]
    # get all countries for that year
    df_country_grouped = df_country[df_country['year'] == year].groupby('cowc_dest').first().reset_index()
    # merge
    #return None
    merger = Merger(df_country_grouped, df_fdi_filtered)
    X = merger.get_result()
    model = keras.models.load_model("../data/model/lstm.keras")
    y = model.predict(X)
    X['label'] = y
    X.sort_values('label', inplace=True)

    return X


df_fdi = pd.read_csv("../data/preprocessed/preprocessed_fdi.csv")
df_fdi['isin'] = df_fdi['isin'].astype(str)
df_country = pd.read_csv("../data/preprocessed/preprocessed_country.csv")

st.write("""## Investment Location Recommendersystem""")

year = st.selectbox('Select a Year', options=sorted(df_fdi['year'].unique()))
available_cowc_source = df_country['cowc_source'].unique()
df_fdi_year = df_fdi[df_fdi['year'] == year]
df_fdi_year_cowc_source = df_fdi_year[df_fdi_year['cowc_source'].isin(available_cowc_source)]
isin = st.selectbox('Enter an ISIN', options=sorted(df_fdi_year_cowc_source['isin'].unique()))

# Get Recommendation Button
#if st.button('Get Recommendation'):
#    recommendation = get_recommendation(isin, year)
#    st.dataframe(recommendation.head())

if __name__ == '__main__':

    g = get_recommendation("GB00B0MSY877", 2006)
