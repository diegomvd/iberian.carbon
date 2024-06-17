import pandas as pd 
import geopandas as gpd
from subprocess import call
from pathlib import Path
import glob
import re
import os


def get_plot_coordinates(plot_id,coord_type):
    #TODO: instead of returning 0 handle with switch statement and exception
    ret = 0
    if coord_type == "latitude":
        ret = ifn4_df[ifn4_df["Estadillo"]==plot_id].reset_index()["CoorY"][0]
    if coord_type == "longitude":
        ret = ifn4_df[ifn4_df["Estadillo"]==plot_id].reset_index()["CoorX"][0]
    return ret

tmpdir = "./IFN_4_SP/tmp/" 
if not Path(tmpdir).exists():
    Path(tmpdir).mkdir(parents=True)

ifn4_files = [file for file in Path("./IFN_4_SP/").glob('*') if "Ifn4" in str(file)]
Sig_files = [file for file in Path("./IFN_4_SP/").glob('*') if "Sig_" in str(file)]

for accdb_ifn4 in ifn4_files:
    
    region = re.findall("_(.*)",accdb_ifn4.stem)[0]
    
    ifn4_out_file = "{}PCDatosMAP_{}.csv".format(tmpdir,region)
    mdb_command = "mdb-export {} PCDatosMap > {}".format(accdb_ifn4,ifn4_out_file)
    os.system(mdb_command)
    
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
            "Estadillo" : "plot_id",
            "CA" : "AGC",
            "CR" : "BGC",
            "BA" : "AGB",
            "BR" : "BGB"
        }
    )

    ifn4_df = pd.read_csv(ifn4_out_file)
    ifn4_df = ifn4_df[["Estadillo","CoorX","CoorY"]]
    
    plot_df["longitude"] = plot_df["plot_id"].apply(lambda x: get_plot_coordinates(x,"longitude"))
    plot_df["latitude"] = plot_df["plot_id"].apply(lambda x: get_plot_coordinates(x,"latitude"))

    plot_gdf = gpd.GeoDataFrame(
        plot_df,
        geometry= gpd.points_from_xy(x=plot_df.longitude, y=plot_df.latitude)
    )
    plot_gdf.crs = "EPSG:4230"
    
    print(plot_gdf)

    break
