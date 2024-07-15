import pandas as pd 
import geopandas as gpd
from subprocess import call
from pathlib import Path
import glob
import re
import os
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np

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
    
    species_name = spdb[spdb["CODIGO ESPECIE"]==species_code].reset_index()["NOMBRE IFN"][0]
    wd = get_wood_density(species_name,wddb)
    agb = (volume_stem + volume_branches) * wd

    return agb    

def get_species_name(species_code, spdb):
    try:
        species_name =  spdb[spdb["CODIGO ESPECIE"]==species_code].reset_index()["NOMBRE IFN"][0]
    except:
        species_name = None
    return species_name   

def get_family(species_name, wddb):
    try:
        if species_name in wddb["Binomial"].values:
            family = wddb[wddb["Binomial"]==species_name].reset_index()["Family"][0]
        else:
            genus = species_name.split(' ')[0]
            wddb2 = wddb.copy()
            wddb2['Binomial'] = wddb2['Binomial'].apply(lambda name: name.split(' ')[0])
            if genus in wddb2["Binomial"].values:
                family = wddb2[wddb2["Binomial"]==genus].reset_index()["Family"][0]
            else:
                family = None    
    except:
        family = None

    return family

def get_type(family_name, db):
    if family_name  is None:
        ret = None
    elif family_name in db['Family'].values:
        ret = 'Gymnosperm'
    else:
        ret = 'Angiosperm'
    return ret

def get_plots_dominant_type(plot_df):
    # TODO: it might be possible to get this done in a vecotrized way avoiding looping (maybe a complex groupby operation)
    type_df = pd.DataFrame()
    for name, group in plot_df.groupby(['Estadillo','Provincia']):
        dominant_type = group[ group.AGB == group.AGB.max()].reset_index()['Species_type'][0] 
        this = pd.DataFrame.from_dict({'Estadillo':[name[0]], 'Provincia':[name[1]], 'Type': [dominant_type]})
        type_df = pd.concat([type_df,this],axis='rows')
    return type_df    


tmpdir = "./IFN_4_SP/tmp/" 
if not Path(tmpdir).exists():
    Path(tmpdir).mkdir(parents=True)

ifn4_files = [file for file in Path("./IFN_4_SP/").glob('*') if "Ifn4" in str(file)]
Sig_files = [file for file in Path("./IFN_4_SP/").glob('*') if "Sig_" in str(file)]

ifn_gdf_28 = gpd.GeoDataFrame()
ifn_gdf_29 = gpd.GeoDataFrame()
ifn_gdf_30 = gpd.GeoDataFrame()
ifn_gdf_31 = gpd.GeoDataFrame()

wood_density_db = pd.read_excel("GlobalWoodDensityDatabase.xls",sheet_name='Data')
gymnosperm_families_db = pd.read_csv('gymnosperm_families.csv')
ifn_species_codes = pd.read_csv("CODIGOS_IFN.csv",delimiter=";").dropna()
ifn_species_codes["CODIGO ESPECIE"] = ifn_species_codes["CODIGO ESPECIE"].apply(lambda c: int(c))

for accdb_ifn4 in ifn4_files:

    region = re.findall("_(.*)",accdb_ifn4.stem)[0]
    
    ifn4_out_file = "{}PCDatosMAP_{}.csv".format(tmpdir,region)
    mdb_command = "mdb-export {} PCDatosMap > {}".format(accdb_ifn4,ifn4_out_file)
    os.system(mdb_command)

    ifn4_pc_parcelas = "{}PCParcelas_{}.csv".format(tmpdir,region)
    mdb_command = "mdb-export {} PCParcelas > {}".format(accdb_ifn4,ifn4_pc_parcelas)
    os.system(mdb_command)

    ifn4_pc_esparc = "{}PCEspParc_{}.csv".format(tmpdir,region)
    mdb_command = "mdb-export {} PCEspParc > {}".format(accdb_ifn4,ifn4_pc_esparc)
    os.system(mdb_command)
    
    if (region == "Canarias") or (region == "Baleares"):
        continue
    # if not region in ['Soria','ACoruna']:
    # if not region in ['Soria']:
    #     continue

    try:
            
        accdb_sig = [file for file in Sig_files if region in str(file)][0]

        sig_out_file = "{}Parcelas_exs_{}.csv".format(tmpdir,region)
        mdb_command = "mdb-export {} Parcelas_exs > {}".format(accdb_sig,sig_out_file)
        os.system(mdb_command)

        plot_df = pd.read_csv(sig_out_file)

        # In some provinces and CCAA biomass and carbon storage is not calculated but Woody Volume is reported. AGB is estimated using the Global Wood Density database to transform volume values to biomass.  
        if not "CA" in list(plot_df.columns):

            plot_df = plot_df[["Estadillo","VLE","VCC","Especie","Provincia"]]
            
            # Estimate AGB and then AGC using a 0.5 factor.
            plot_df["AGB"] = plot_df.apply(lambda x: compute_biomass(x["VCC"], x["VLE"], x["Especie"], wood_density_db, ifn_species_codes), axis=1)
            plot_df["AGC"] = plot_df["AGB"] * 0.5

            plot_df['id'] = [val for val in zip(plot_df["Estadillo"],plot_df["Provincia"])]

            na_id = plot_df[plot_df["AGB"].isnull()]["id"].unique()
            plot_df = plot_df[ ~plot_df["id"].isin(na_id) ]
                        
            # Leave belowground stocks empty.            
            plot_df["BGB"] = np.nan
            plot_df["BGC"] = np.nan
            
            plot_df = plot_df[["Estadillo","AGC","BGC","AGB","BGB","Especie","Provincia"]]

        else:

            plot_df = plot_df[["Estadillo","CA","CR","BA","BR","Especie","Provincia"]]
         
            plot_df = plot_df.rename(
                columns = {
                    "Estadillo" : "Estadillo",
                    "CA" : "AGC",
                    "CR" : "BGC",
                    "BA" : "AGB",
                    "BR" : "BGB",
                    "Especie":"Especie",
                    "Provincia":"Provincia"
                }
            )    

        # TODO: Concentrate everything in a single function transforming IFN code to type.
        plot_df['Species_name'] = plot_df.apply(lambda x: get_species_name(x['Especie'],ifn_species_codes),axis=1)
        plot_df['Species_family'] = plot_df.apply(lambda x: get_family(x['Species_name'],wood_density_db),axis=1)
        plot_df['Species_type'] = plot_df.apply(lambda x: get_type(x['Species_family'],gymnosperm_families_db),axis=1)

        # Get dominant tree type for each NFI plot.
        biomass_per_type = plot_df.groupby(['Estadillo','Species_type','Provincia']).agg({ 'AGB':'sum', 'AGC':'sum', 'BGB':'sum', 'BGC':'sum' }).reset_index()
        plot_types = get_plots_dominant_type(biomass_per_type)   

        # Drop the species information.
        plot_df = plot_df[["Estadillo","AGC","BGC","AGB","BGB","Provincia"]]

        # Sum storage contribution for every species and diameter class.
        plot_df = plot_df.groupby(["Estadillo","Provincia"]).sum().reset_index()

        plot_df['id'] = [val for val in zip(plot_df["Estadillo"],plot_df["Provincia"])]

        # Include dominant type information in the stocks dataset.
        plot_types['id'] = [val for val in zip(plot_types['Estadillo'],plot_types['Provincia'])]
        plot_types = plot_types.drop(columns=['Estadillo','Provincia'])
        plot_df = plot_df.join(plot_types.set_index("id"),on="id",how='inner')
        
        # Include UTM and coordinates information in the stocks dataset.
        ifn4_df = pd.read_csv(ifn4_out_file)
        ifn4_df = ifn4_df[["Estadillo","CoorX","CoorY",'Huso','Provincia']]
        ifn4_df['id'] = [val for val in zip(ifn4_df["Estadillo"],ifn4_df["Provincia"])]
        ifn4_df = ifn4_df.drop(columns=['Estadillo','Provincia'])
        plot_df = plot_df.join(ifn4_df.set_index("id"),on="id",how='inner')

        # Include date information in the stocks dataset.
        ifn4_date_df = pd.read_csv(ifn4_pc_parcelas)
        ifn4_date_df = ifn4_date_df[["Estadillo","FechaIni","Provincia"]]
        ifn4_date_df['id'] = [val for val in zip(ifn4_date_df["Estadillo"],ifn4_date_df["Provincia"])]
        ifn4_date_df = ifn4_date_df.drop(columns=['Estadillo','Provincia'])
        plot_df = plot_df.join(ifn4_date_df.set_index("id"),on="id",how='inner')
        
        # Retain only the year.
        plot_df['Year']=plot_df['FechaIni'].apply(lambda date: "20{}".format(date[6:8]))

        # Include province or CCAA information.
        plot_df['Region'] = region

        # Iterate over UTMs to build one dataset per each.
        for utm in plot_df['Huso'].unique():

            if utm is None:
                continue
            utm = int(utm)

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
            plot_gdf = plot_gdf.drop(columns=['CoorX','CoorY','Huso','FechaIni','id'])

            match utm:
                case 28:
                    ifn_gdf_28 = pd.concat([ifn_gdf_28,plot_gdf],axis='rows')
                case 29:
                    ifn_gdf_29 = pd.concat([ifn_gdf_29,plot_gdf],axis='rows')
                case 30:
                    ifn_gdf_30 = pd.concat([ifn_gdf_30,plot_gdf],axis='rows')
                case 31:
                    ifn_gdf_31 = pd.concat([ifn_gdf_31,plot_gdf],axis='rows')            
                    
    except:
        continue    

ifn_gdf_28 = ifn_gdf_28.reset_index()
ifn_gdf_29 = ifn_gdf_29.reset_index()
ifn_gdf_30 = ifn_gdf_30.reset_index()
ifn_gdf_31 = ifn_gdf_31.reset_index()

savedir = "ifn4_biomass/"
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