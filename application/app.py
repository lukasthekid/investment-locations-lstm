import streamlit as st
import pandas as pd
import os

# sys.path.insert(0, "../scripts")

#from application.utils import Merger
from scripts.utils import Merger
from scripts.utils import get_country_name
from tensorflow import keras

df_fdi = pd.read_csv("../data/preprocessed/preprocessed_fdi.csv")
df_fdi['isin'] = df_fdi['isin'].astype(str)
df_country = pd.read_csv("../data/preprocessed/preprocessed_country.csv")
df_model = pd.read_csv("../data/preprocessed/preprocessed_data.csv")

def get_recommendation(year:int, country:str, isin:str):
    #print(os.getcwd())
    # get the company from the FDI dataframe
    df_fdi_filtered = df_fdi[(df_fdi['year'] == year) & (df_fdi['cowc_source'] == country) & (df_fdi['isin'] == isin)].iloc[0:1]
    # get all countries the companys source country invested to in that year
    df_country_grouped = df_country[(df_country['year'] == year) & (df_country['cowc_source'] == country)]\
        .groupby('cowc_dest').first().reset_index()

    merger = Merger(df_country=df_country_grouped, df_fdi=df_fdi_filtered)
    X = merger.get_result()
    df_model_X = df_model.drop(['invest_actual'], axis=1)
    # get all the columns from the model data frame
    cols = set(df_model_X.columns) - set(X.columns)
    for col in cols:
        X[col] = 0
    #only take common columns in same order
    X = X[df_model_X.columns]
    model = keras.models.load_model("../data/model/lstm.keras")
    X.to_csv("../data/test_str.csv")
    X_test = X.values.reshape((X.shape[0], 1, X.shape[1]))
    y = model.predict(X_test)
    X['label'] = y

    return transform_output(X)

def transform_output(df_raw:pd.DataFrame):
    # Backtransform 'cowc_dest' one-hot encoded columns
    cols = [col for col in df_raw.columns if 'cowc_dest' in col]
    df_subset = df_raw[cols]
    # Back-transform the one-hot encoded values
    df_raw['Country'] = df_subset.idxmax(axis=1).str.replace('cowc_dest_', '')
    df = df_raw.sort_values('label', ascending=False).reset_index(drop=True)
    #df['Country'] = df.apply(lambda x: get_country_name(str(df['Country'])))
    return df[['Country', 'label']].head()



st.write("""## Investment Location Recommendersystem""")
#year select box
year = st.selectbox('Select a Year', options=sorted(df_fdi['year'].unique()))
if year:
    # select the source country
    df_fdi_year = df_fdi[df_fdi['year'] == year]
    available_cowc_source = df_fdi_year['cowc_source'].unique()
    country = st.selectbox("Select the Company's base Country", options=sorted(available_cowc_source))

    if country:
        # select the company you want the predictions for
        df_fdi_year_country = df_fdi_year[df_fdi_year['cowc_source'] == country]
        available_isin = df_fdi_year_country['isin'].unique()
        isin = st.selectbox('Select an ISIN', options=sorted(available_isin))

if st.button('Get Recommendations'):
    res = get_recommendation(year, country, isin)
    st.dataframe(res.head())





df_fdi_year = df_fdi[df_fdi['year'] == year]
df_fdi_year_cowc_source = df_fdi_year[df_fdi_year['cowc_source'].isin(available_cowc_source)]


# Get Recommendation Button
#if st.button('Get Recommendation'):
#    recommendation = get_recommendation(isin, year)
#    st.dataframe(recommendation.head())

if __name__ == '__main__':

    g = get_recommendation(2009, 'DE', 'DE0005297204')
    print(g.head())
