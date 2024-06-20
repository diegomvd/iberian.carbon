from geoutils import reproject_raster
from pathlib import Path

target_proj = 'EPSG:25830'

savedir = "PNOA2_LiDAR_Vegetation_25m_ETRS89_UTM30N/"
if not Path(savedir).exists():
    Path(savedir).mkdir(parents=True)

for fname in Path("PNOA2_LiDAR_Vegetation_25m/").glob('NDSM*.tif'):

    name = fname.stem
    suffix = fname.suffix
    out = "{}{}{}".format(savedir,name,suffix)
    out_file = reproject_raster(target_projection_crs=target_proj,base_raster_file=fname,target_raster_file=out)