import geopandas as gpd 
import pandas as pd
from pathlib import Path

country_scale_df = pd.DataFrame()

stocks_dir = '/Users/diegobengochea/git/iberian.carbon/data/stocks_NFI4/'
for stocks_file in Path(stocks_dir).glob('*foresttype.shp'):
    stocks = gpd.read_file(stocks_file)
    stocks['BGB_Ratio'] = stocks['BGB']/stocks['AGB']
    country_scale_df = pd.concat([country_scale_df,pd.DataFrame(stocks.drop(columns='geometry'))],axis='rows')

country_scale_df.to_csv('BGBRatios_raw.csv')
