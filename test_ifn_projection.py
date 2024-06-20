import geopandas as gpd

print("start")
ifn = gpd.read_file('ifn4_filtered.shp')
print(ifn)
ifn = ifn.to_crs("EPSG:4326")
print(ifn)