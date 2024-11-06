from geoutils import reproject_raster
from pathlib import Path

dir_PNOA = '/Users/diegobengochea/git/iberian.carbon/data/training_data_Sentinel2_PNOA/'
dir_Sentinel2 = '/Users/diegobengochea/git/iberian.carbon/data/Sentinel2_Composites_Spain/'
out_dir = '/Users/diegobengochea/git/iberian.carbon/data/training_data_Sentinel2_PNOA_UTM30/'

if not Path(out_dir).exists():
    Path(out_dir).mkdir(parents=True)

print('Reprojecting:')

for dir in [dir_PNOA,dir_Sentinel2]:
    for i, file in enumerate(Path(dir).rglob('*.tif')):

        print('file {}'.format(i))

        output_file = '{}{}_epsg_25830.tif'.format(out_dir,file.stem) 

        reproject_raster('EPSG:25830',file,output_file)
print('Done.')
