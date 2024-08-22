from geoutils import reproject_raster
from pathlib import Path

dir = '/Users/diegobengochea/git/iberian.carbon/data/*/'
out_dir = '/Users/diegobengochea/git/iberian.carbon/data/*/'

if not Path(out_dir).exists():
    Path(out_dir).mkdir(parents=True)

print('Reprojecting:')
for i, file in enumerate(Path(dir).glob('*.tif')):

    print('file {}'.format(i))

    output_file = '{}{}_epsg_25830.tif'.format(out_dir,file.stem) 

    reproject_raster('EPSG:25830',file,output_file)
print('Done.')