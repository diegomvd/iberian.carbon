import pandas as pd 
import geopandas as gpd
from subprocess import call
from pathlib import Path
import glob
import re
import os
import seaborn as sns
from matplotlib import pyplot as plt

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
        return epsg_ed50[utmzone]
    elif region == 'Canarias':
        return 'EPSG:4326'
    else:
        return epsg_etrs89[utmzone]       


tmpdir = "./IFN_4_SP/tmp/" 
if not Path(tmpdir).exists():
    Path(tmpdir).mkdir(parents=True)

ifn4_files = [file for file in Path("./IFN_4_SP/").glob('*') if "Ifn4" in str(file)]
Sig_files = [file for file in Path("./IFN_4_SP/").glob('*') if "Sig_" in str(file)]

ifn_gdf_28 = gpd.GeoDataFrame()
ifn_gdf_29 = gpd.GeoDataFrame()
ifn_gdf_30 = gpd.GeoDataFrame()
ifn_gdf_31 = gpd.GeoDataFrame()

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

    try:
        
        accdb_sig = [file for file in Sig_files if region in str(file)][0]
    
        sig_out_file = "{}Parcelas_exs_{}.csv".format(tmpdir,region)
        mdb_command = "mdb-export {} Parcelas_exs > {}".format(accdb_sig,sig_out_file)
        os.system(mdb_command)

        plot_df = pd.read_csv(sig_out_file)

        if not "CA" in list(plot_df.columns):
            continue

        plot_df = plot_df[["Estadillo","CA","CR","BA","BR"]]
        plot_df = plot_df.rename(
            columns = {
                "Estadillo" : "Estadillo",
                "CA" : "AGC",
                "CR" : "BGC",
                "BA" : "AGB",
                "BR" : "BGB"
            }
        )

        ifn4_df = pd.read_csv(ifn4_out_file)
        ifn4_df = ifn4_df[["Estadillo","CoorX","CoorY",'Huso']]
        
        plot_df = plot_df.join(ifn4_df.set_index("Estadillo"),on="Estadillo")

        ifn4_date_df = pd.read_csv(ifn4_pc_parcelas)
        ifn4_date_df = ifn4_date_df[["Estadillo","FechaIni"]]

        plot_df = plot_df.join(ifn4_date_df.set_index("Estadillo"),on="Estadillo")

        ifn4_state_df = pd.read_csv(ifn4_pc_esparc)
        ifn4_state_df = ifn4_state_df[["Estadillo","Estado","FPMasa","Edad"]]

        plot_df = plot_df.join(ifn4_state_df.set_index("Estadillo"),on="Estadillo")

        plot_df['year']=plot_df['FechaIni'].apply(lambda date: "20{}".format(date[6:8]))

        plot_df['region'] = region

        plot_gdf = gpd.GeoDataFrame(
            plot_df,
            geometry= gpd.points_from_xy(x=plot_df.CoorX, y=plot_df.CoorY)
        )

        utmzone = plot_df['Huso'].unique()[0]
        crs = get_region_UTM(region,str(utmzone))
        
        plot_gdf = plot_gdf.set_crs(crs)
        # plot_gdf = plot_gdf.to_crs('EPSG:25830')

        plot_gdf = plot_gdf.drop(columns=['CoorX','CoorY','Huso','FechaIni'])

        match utmzone:
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

# sns.displot(ifn_gdf,x='year',kind='hist')
# plt.show()

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