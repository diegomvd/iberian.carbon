from geoutils import downsample_raster
from pathlib import Path

target_res = 25

savedir = "PNOA2_LiDAR_Vegetation_25m/"
if not Path(savedir).exists():
    Path(savedir).mkdir(parents=True)

for fname in Path("PNOA2_LIDAR_vegetation/").glob('NDSM*.tif'):

    name = fname.stem
    suffix = fname.suffix
    out = "{}{}{}".format(savedir,name,suffix)
    out_file = downsample_raster(target_res,fname,out)