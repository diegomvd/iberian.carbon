import pandas as pd     
from  sklearn.metrics import r2_score, mean_absolute_error
from scipy.optimize import curve_fit
import numpy as np
from scipy import stats
from sklearn.linear_model import QuantileRegressor


def fit_allometry(allometry_df,df,h,agb,ftype,tier):
    regression_mean = stats.linregress(np.log(df[h]), np.log(df[agb]))
    regression_q05 = QuantileRegressor(quantile=0.05,alpha=0.001).fit(np.log(df[h].to_numpy().reshape(-1, 1)), np.log(df[agb].to_numpy()))
    regression_q95 = QuantileRegressor(quantile=0.95,alpha=0.001).fit(np.log(df[h].to_numpy().reshape(-1, 1)), np.log(df[agb].to_numpy()))

    row = pd.DataFrame({
        'ForestType':ftype,
        'Tier': tier,
        'Intercept_mean':[regression_mean.intercept],
        'Slope_mean':[regression_mean.slope],
        'Intercept_q05':[regression_q05.intercept_],
        'Slope_q05':regression_q05.coef_,
        'Intercept_q95':[regression_q95.intercept_],
        'Slope_q95':regression_q95.coef_,
        'Samples':[len(df.index)]
    })
    allometry_df = pd.concat([allometry_df,row],axis='rows')
    return allometry_df

def get_tier_class(forest_types,mfe_class,tier):
    ret = forest_types[forest_types['ForestTypeMFE']==mfe_class].reset_index().at[0,tier]
    return ret

def input_type_tier(h_agb_df,forest_types,tier):
    h_agb_df[tier]=h_agb_df.apply(lambda row: get_tier_class(forest_types,row.ForestType,tier), axis=1) 
    return h_agb_df   


#############################################

height_biomass_df = pd.read_csv('/home/diego/git/iberian.carbon/data/HeightBiomassTable.csv')
p99_height_max = np.percentile(height_biomass_df['Hmean'],99)
p01_height_max = np.percentile(height_biomass_df['Hmean'],1)
height_biomass_df = height_biomass_df[(height_biomass_df['Hmean']<p99_height_max) & (height_biomass_df['Hmean']>p01_height_max) ]

forest_types = pd.read_csv('/home/diego/git/iberian.carbon/data/all_forest_types.csv')
tiers=['Clade','Family','Genus']
for tier in tiers:
    height_biomass_df=input_type_tier(height_biomass_df,forest_types,tier)

#############################################

allometry_df = pd.DataFrame()

df = height_biomass_df.copy()
if len(df.index)>10:    
    allometry_df=fit_allometry(allometry_df,df,'Hmean','AGB','General',0)

    for clade in height_biomass_df['Clade'].unique():
        df = height_biomass_df[height_biomass_df['Clade']==clade]
        if len(df.index)>10 and not clade in allometry_df.ForestType.unique():
            allometry_df=fit_allometry(allometry_df,df,'Hmean','AGB',clade,1)

        for family in height_biomass_df['Family'].unique():
            df = height_biomass_df[(height_biomass_df['Family']==family) & (height_biomass_df['Clade']==clade)]
            if len(df.index)>10 and not family in allometry_df.ForestType.unique():
                allometry_df=fit_allometry(allometry_df,df,'Hmean','AGB',family,2)

            for genus in height_biomass_df['Genus'].unique():
                df = height_biomass_df[(height_biomass_df['Genus']==genus) & (height_biomass_df['Family']==family) & (height_biomass_df['Clade']==clade)]
                if len(df.index)>10 and not genus in allometry_df.ForestType.unique():
                    allometry_df=fit_allometry(allometry_df,df,'Hmean','AGB',genus,3) 

                for mfe_class in height_biomass_df['ForestType'].unique():
                    df = height_biomass_df[(height_biomass_df['ForestType']==mfe_class) & (height_biomass_df['Genus']==genus) & (height_biomass_df['Family']==family) & (height_biomass_df['Clade']==clade)]
                    if len(df.index)>10 and not mfe_class in allometry_df.ForestType.unique():
                        allometry_df=fit_allometry(allometry_df,df,'Hmean','AGB',mfe_class,4)      

allometry_df.to_csv('data/H_AGB_Allometries_Tiers.csv',index=False)
