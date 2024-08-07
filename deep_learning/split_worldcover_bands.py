from geoutils import reproject_raster
from pathlib import Path

dir = '/Users/diegobengochea/git/iberian.carbon/data/*/'
out_dir = '/Users/diegobengochea/git/iberian.carbon/data/*/'

if not Path(out_dir).exists():
    Path(out_dir).mkdir(parents=True)

print('Splitting bands')
for i, file in enumerate(Path(dir).glob('*.tif')):

    print('file {}'.format(i))

    with rasterio.open(file) as src:
        for band in range(1, src.count):
            output_file = '{}{}_B0{}.tif'.format(out_dir,file.stem,band)

            out_meta = src.meta.copy()

            out_meta.update({"count": 1})

            # save the clipped raster to disk
            with rasterio.open(output_file, "w", **out_meta) as dest:
                dest.write(src.read(band),1) 

print('Done.')