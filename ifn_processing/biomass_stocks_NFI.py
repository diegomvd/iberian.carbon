import geopandas as gpd
from pathlib import Path
import re
import glob
import os
import pandas as pd
import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

"""
Functions declaration
"""
def get_region_UTM(region,utmzone):
    
    list_ED50 = [
        'Navarra',
        'ACoruna',
        'Lugo',
        'Orense',
        'Pontevedra',
        'Asturias',
        'Cantabria',
        'Murcia',
        'Baleares', 
        'Euskadi',
        'LaRioja',
        'Madrid',
        'Cataluna'
    ]

    epsg_ed50 = {'28':'EPSG:23028','29':'EPSG:23029','30':'EPSG:23030','31':'EPSG:23031'}
    epsg_etrs89 = {'28':'EPSG:25828','29':'EPSG:25829','30':'EPSG:25830','31':'EPSG:25831'}
    
    if region in list_ED50:
        return epsg_ed50[utmzone], epsg_etrs89[utmzone] 
    elif region == 'Canarias':
        return 'EPSG:4326', 'EPSG:25828'
    else:
        return epsg_etrs89[utmzone], epsg_etrs89[utmzone]        

def get_species_name(species_code, spdb):
    try:
        species_name =  spdb[spdb["CODIGO ESPECIE"]==species_code].reset_index()["NOMBRE IFN"][0]
    except:
        species_name = None
    return species_name 

def get_wood_density(species_name, wddb):
    if species_name in wddb["Binomial"].values:
        wd = wddb[wddb["Binomial"]==species_name]["Wood density (g/cm^3), oven dry mass/fresh volume"].dropna().mean()
    else: 
        genus = species_name.split(' ')[0]
        wddb2 = wddb.copy()
        wddb2['Binomial'] = wddb2['Binomial'].apply(lambda name: name.split(' ')[0])
        if genus in wddb2["Binomial"].values:
            wd = wddb2[wddb2["Binomial"]==genus]["Wood density (g/cm^3), oven dry mass/fresh volume"].dropna().mean()
        else:
            wd = wddb["Wood density (g/cm^3), oven dry mass/fresh volume"].dropna().mean()
    return wd        

def compute_biomass(volume_stem, volume_branches, species_code, wddb, spdb):
    
    species_name = get_species_name(species_code,spdb)
    try:
        wd = get_wood_density(species_name,wddb)
        agb = (volume_stem + volume_branches) * wd
    except:
        agb = np.nan    

    return agb    

"""
Main
"""

tmpdir = "/Users/diegobengochea/git/iberian.carbon/data/IFN_4_SP/tmp/" 
if not Path(tmpdir).exists():
    Path(tmpdir).mkdir(parents=True)

ifn4_files = [file for file in Path("/Users/diegobengochea/git/iberian.carbon/data/IFN_4_SP/").glob('Ifn4*')]
Sig_files = [file for file in Path("/Users/diegobengochea/git/iberian.carbon/data/IFN_4_SP/").glob('Sig_*')]

ifn_gdf_28 = gpd.GeoDataFrame()
ifn_gdf_29 = gpd.GeoDataFrame()
ifn_gdf_30 = gpd.GeoDataFrame()
ifn_gdf_31 = gpd.GeoDataFrame()

wood_density_db = pd.read_excel("/Users/diegobengochea/git/iberian.carbon/data/GlobalWoodDensityDatabase.xls",sheet_name='Data')
ifn_species_codes = pd.read_csv("/Users/diegobengochea/git/iberian.carbon/data/CODIGOS_IFN.csv",delimiter=";").dropna()
ifn_species_codes["CODIGO ESPECIE"] = ifn_species_codes["CODIGO ESPECIE"].apply(lambda c: int(c))

"""
Iterating over NFI database files for each province. These files contain general information about NFI plots.
Actual information on biomass stocks can be found in the Sig_files, within Parcelas_exs table, where they are 
stratified by species and diametric class.  
"""
for accdb_ifn4 in ifn4_files:

    region = re.findall("_(.*)",accdb_ifn4.stem)[0]
    logger.info(f"Processing province {region}.")
    
    ifn4_out_file = "{}PCDatosMAP_{}.csv".format(tmpdir,region)
    if not Path(ifn4_out_file).exists():
        mdb_command = "mdb-export {} PCDatosMap > {}".format(accdb_ifn4,ifn4_out_file)
        os.system(mdb_command)

    ifn4_pc_parcelas = "{}PCParcelas_{}.csv".format(tmpdir,region)
    if not Path(ifn4_pc_parcelas).exists():
        mdb_command = "mdb-export {} PCParcelas > {}".format(accdb_ifn4,ifn4_pc_parcelas)
        os.system(mdb_command)

    ifn4_pc_esparc = "{}PCEspParc_{}.csv".format(tmpdir,region)
    if not Path(ifn4_pc_esparc).exists():
        mdb_command = "mdb-export {} PCEspParc > {}".format(accdb_ifn4,ifn4_pc_esparc)
        os.system(mdb_command)
    
    if (region == "Canarias") or (region == "Baleares"):
        continue

    """
    Try to compute biomass stocks in the current province.
    """    
    logger.info("Attempting computation of total biomass stocks.")
    try:

        # Get the SIG file corresponding to the current region.     
        accdb_sig = [file for file in Sig_files if region in str(file)][0]

        # Load the Parcelas_exs table containing stocks per species and diameter class
        sig_out_file = "{}Parcelas_exs_{}.csv".format(tmpdir,region)
        if not Path(sig_out_file).exists():
            mdb_command = "mdb-export {} Parcelas_exs > {}".format(accdb_sig,sig_out_file)
            os.system(mdb_command)
        plot_df = pd.read_csv(sig_out_file)

        if not len(plot_df.index)>0:
            logger.error('Empty stocks table.')

        """
        In some provinces and CCAA biomass and carbon storage is not calculated but Woody Volume is reported. AGB is estimated using the Global Wood Density database to transform volume values to biomass. Every stocks table that does not contain the entry CA (Aboveground Carbon), is treated within the conditional section below.
        """
        if not "CA" in list(plot_df.columns):

            logger.info('Biomass stocks per diameter class and species are not reported, calculating from woody volume...')

            # Filter out columns to retain only the most relevant ones: plot_id, province, wood volume and species.
            plot_df = plot_df[["Estadillo","VLE","VCC","Especie","Provincia"]]
            
            # Estimate AGB from the woody volume the species and using the global wood density table. 
            plot_df["AGB"] = plot_df.apply(lambda x: compute_biomass(x["VCC"], x["VLE"], x["Especie"], wood_density_db, ifn_species_codes), axis=1)

            # Index plots by the tuple (Plot_id, Province) because plot_ids are repeated across provinces and remove NoData plots.
            plot_df['Index'] = [val for val in zip(plot_df["Estadillo"],plot_df["Provincia"])]
            # Remove NoData stocks.
            na_id = plot_df[plot_df["AGB"].isnull()]["Index"].unique()
            plot_df = plot_df[ ~plot_df["Index"].isin(na_id) ]
                        
            # Leave belowground stocks empty.            
            plot_df["BGB"] = np.nan
            
            # Filter relevant columns: Plot_id, Stocks and Province
            plot_df = plot_df[["Index","AGB","BGB"]]
            logger.info('...done.')

        else:
            """
            In this conditional group the plots with reported biomass values are treated
            """
            logger.info('Biomass stocks per diameter class and species are reported. Retrieving data...')
            # Filter relevant columns: Plot_id, Stocks and Province
            plot_df = plot_df[["Estadillo","Provincia","BA","BR"]]

            # Index plots by the tuple (Plot_id, Province) because plot_ids are repeated across provinces.
            plot_df['Index'] = [val for val in zip(plot_df["Estadillo"],plot_df["Provincia"])]
         
            # Rename columns for coherence
            plot_df = plot_df.rename(
                columns = {
                    "Index" : "Index",
                    "BA" : "AGB",
                    "BR" : "BGB",
                }
            )    

            plot_df = plot_df[["Index","AGB","BGB"]]

            logger.info('...done.')

        logger.info('Summing total biomass across species and diameter classes...')
        # Sum storage contribution for every species and diameter class in each plot. 
        plot_df = plot_df.groupby("Index").sum().reset_index()
        logger.info('..done')
    
        """
        Now include spatiotemporal information about the plot data: UTM, location, and date. 
        """
        logger.info('Add spatio-temporal information to the dataset...')
        # Include UTM and coordinates information in the stocks dataset.
        ifn4_df = pd.read_csv(ifn4_out_file)
        ifn4_df = ifn4_df[["Estadillo","CoorX","CoorY",'Huso','Provincia']]
        ifn4_df['Index'] = [val for val in zip(ifn4_df["Estadillo"],ifn4_df["Provincia"])]
        ifn4_df = ifn4_df.drop(columns=['Estadillo','Provincia'])
        
        plot_df = plot_df.join(ifn4_df.set_index("Index"),on="Index",how='inner')

        # Include date information in the stocks dataset.
        ifn4_date_df = pd.read_csv(ifn4_pc_parcelas)
        ifn4_date_df = ifn4_date_df[["Estadillo","FechaIni","Provincia"]]
        ifn4_date_df['Index'] = [val for val in zip(ifn4_date_df["Estadillo"],ifn4_date_df["Provincia"])]
        ifn4_date_df = ifn4_date_df.drop(columns=['Estadillo','Provincia'])
        
        plot_df = plot_df.join(ifn4_date_df.set_index("Index"),on="Index",how='inner')

        # Retain only the year.
        plot_df['Year']=plot_df['FechaIni'].apply(lambda date: "20{}".format(date[6:8]))

        # Include province or CCAA information.
        plot_df['Region'] = region

        logger.info('...done.')

        logger.info('Store results from processed region.')
        # Iterate over UTMs to build one dataset per each.
        for utm in plot_df['Huso'].unique():

            if utm is None:
                continue
            utm = int(utm)

            # Filter out observations from a different UTM.
            plot_df_filtered = plot_df[plot_df['Huso']==utm]

            # Create a georeferenced dataset.    
            plot_gdf = gpd.GeoDataFrame(
                plot_df_filtered,
                geometry= gpd.points_from_xy(x=plot_df_filtered.CoorX, y=plot_df_filtered.CoorY)
            )   

            # Ensure the ETRS1989 is used across datasets instead of ED50.
            current_crs, target_crs = get_region_UTM(region,str(utm))           
            plot_gdf = plot_gdf.set_crs(current_crs)
            plot_gdf = plot_gdf.to_crs(target_crs)
            
            # Drop unnecesary columns.
            plot_gdf = plot_gdf.drop(columns=['CoorX','CoorY','Huso','FechaIni'])

            match utm:
                case 28:
                    ifn_gdf_28 = pd.concat([ifn_gdf_28,plot_gdf],axis='rows')
                case 29:
                    ifn_gdf_29 = pd.concat([ifn_gdf_29,plot_gdf],axis='rows')
                case 30:
                    ifn_gdf_30 = pd.concat([ifn_gdf_30,plot_gdf],axis='rows')
                case 31:
                    ifn_gdf_31 = pd.concat([ifn_gdf_31,plot_gdf],axis='rows')            
                    
    except Exception as e:
        logger.error(f"Error in biomass calculation: {str(e)}")
        continue

logger.info('Finished computation, saving results.')
ifn_gdf_28 = ifn_gdf_28.reset_index()
ifn_gdf_29 = ifn_gdf_29.reset_index()
ifn_gdf_30 = ifn_gdf_30.reset_index()
ifn_gdf_31 = ifn_gdf_31.reset_index()

savedir = "/Users/diegobengochea/git/iberian.carbon/data/stocks_NFI4/"
if not Path(savedir).exists():
    Path(savedir).mkdir(parents=True)

if len(ifn_gdf_28.index>0):
    ifn_gdf_28.to_file("{}ifn4_28_biomass.shp".format(savedir),driver='ESRI Shapefile')
if len(ifn_gdf_29.index>0):
    ifn_gdf_29.to_file("{}ifn4_29_biomass.shp".format(savedir),driver='ESRI Shapefile')
if len(ifn_gdf_30.index>0):    
    ifn_gdf_30.to_file("{}ifn4_30_biomass.shp".format(savedir),driver='ESRI Shapefile')
if len(ifn_gdf_31.index>0):
    ifn_gdf_31.to_file("{}ifn4_31_biomass.shp".format(savedir),driver='ESRI Shapefile')

logger.info(f'Results were saved in {savedir}. Terminating program.')