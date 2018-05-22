import glob

import fiona
import matplotlib.pyplot as plt
import os
import numpy as np
from satsense import WORLDVIEW2, SatelliteImage
from satsense.util import load_shapefile2multipolygon, show_multipolygon, multipolygon2mask, save_mask2file
from rasterio import features
from rasterio.features import shapes
from shapely.geometry import shape, MultiPolygon
import geopandas as gpd

image_file = "/home/max/Documents/ai/scriptie/data/17FEB16053453-M2AS_R1C2-056239125020_01_P010.TIF"
bands = WORLDVIEW2
sat_image = SatelliteImage.load_from_file(image_file, bands)

base_path = "/home/max/Documents/ai/scriptie/data/Banglore_slum_type_shape_files/"
pattern = base_path + "*.shp"
for file_name in glob.glob(pattern):
    # file_name = "Tiled_slum"
    if "Road" in file_name or "Temporary" in file_name or "Asbestos_slum" in file_name:
        continue
    print(file_name)

    shape_file_name = file_name
    # shape_file_name = "/home/max/Documents/ai/scriptie/data/Banaglore Slum Details/Banglore approval map/Banglore approval shp files/slums_approved.shp"
    shape_file_name = "/home/max/Documents/ai/scriptie/data/slums_approved.shp"

    # shape_file = base_path + file_name + ".shp"

    GREEN = '#008000'
    offset = 100
    alpha = 0.8

    multi, bounds = load_shapefile2multipolygon(shape_file_name)
    # fp = fiona.open(shape_file_name)
    # print(fp.schema)
    # bounds = fp.bounds
    # shapes = []
    # for pol in fp:
    #     # print(pol)
    #     print(pol['geometry'])
    #     print(pol['geometry']['type'])
    #     shapes.append(shape(pol['geometry']))

    # multipol = MultiPolygon([shape(pol['geometry']) for pol in fp if 'geometry' in pol and pol['geometry'] is not None])
    # fp.close()

    # return multipol, bounds

    xmin, ymin, xmax, ymax = bounds
    extent = int(xmin) - offset, int(ymin) - offset, int(xmax) + offset, int(ymax) + offset
    # _, ax = plt.subplots()
    # ax = show_multipolygon(multi, ax, False, extent, GREEN, alpha,'Rectangles shape multipolygon- toy example')

    rows = sat_image.shape[0]
    cols = sat_image.shape[1]
    # rows = 12
    # cols = 12
    default_val = 255
    fill_val = 0
    # trans = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    # trans = [101985.0, 300.0379266750948, 0.0,
    #      2826915.0, 0.0, -300.041782729805]
    # binary_mask = features.rasterize(
    #     shapes=fp,
    #     default_value = default_val,
    #     fill = fill_val,
    #     out_shape = (rows, cols),
    #     all_touched = False,
    #     dtype = None)

    binary_mask = multipolygon2mask(multi, rows, cols, default_val)

    print(binary_mask)
    print(binary_mask.shape)
    plt.imshow(binary_mask, cmap='gray')
    plt.title('Binary mask')
    print(binary_mask.min())
    plt.show()
    print('Min {} Max {}'.format(binary_mask.min(), binary_mask.max()))
    print('Len > 0: {}'.format(len(np.where(binary_mask > 0)[0])))
    print(np.where(binary_mask > 0))

    # mask_fullfname = "/home/max/Documents/ai/scriptie/data/" + os.path.basename(file_name) + ".tif"
    # save_mask2file(binary_mask, mask_fullfname)

    # fp.close()
    break


