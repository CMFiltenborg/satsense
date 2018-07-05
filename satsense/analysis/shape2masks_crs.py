import fiona
import gdal
import rasterio
from rasterio import mask
from rasterio.features import shapes
import matplotlib.pyplot as plt
import numpy as np
from satsense import SatelliteImage
from satsense.analysis.plot import load_from_file
from satsense.bands import MASK_BANDS, WORLDVIEW3
from satsense.classification.model import get_y_vector
from satsense.generators import CellGenerator
from satsense.image import normalize_image, get_rgb_bands
import cv2

from satsense.performance import jaccard_index_binary_masks

images = [
    # 'section_1',
    'section_2',
    # 'section_3',
]

smallest_window_size = (30, 30)
percentage_threshold = 0.5
base_path = "/home/max/Documents/ai/scriptie/data/Clip"
# shape_file_path = "/home/max/Documents/ai/scriptie/data/Clip/slums_approved.shp"
shape_file_path = "/home/max/Documents/ai/scriptie/data/Clip/slums_approved_manual_edit.shp"

for image_name in images:
    with fiona.open(shape_file_path, "r") as shapefile:
        geoms = [feature["geometry"] for feature in shapefile]

    extension = 'tif'

    image_file = "{base_path}/{image_name}.{extension}".format(
        base_path=base_path,
        image_name=image_name,
        extension=extension
    )

    with rasterio.open(image_file) as src:
        print(src.crs)
        out_image, out_transform = mask.mask(src, geoms, crop=False, invert=False, nodata=0)
        out_meta = src.meta.copy()

    out_meta.update({"driver": "GTiff",
                     "height": out_image.shape[1],
                     "width": out_image.shape[2],
                     "transform": out_transform})

    out_file = "{base_path}/{image_name}_masked.tif".format(
        base_path=base_path,
        image_name=image_name,
    )

    with rasterio.open(out_file, "w", **out_meta) as dest:
        dest.write(out_image)

    dataset = gdal.Open(out_file, gdal.GA_ReadOnly)
    # dataset = dataset[0, :, :]

    array = dataset.ReadAsArray()
    print(array.shape)
    array = np.min(array, 0)
    array = array[:, :, np.newaxis]

    truth_mask = np.where(array > 0, 1, 0)
    # unique, counts = np.unique(array, return_counts=True)
    # median = np.median(unique)
    # array = np.where(array > 0, 1, 0)

    # binary_sat_image = SatelliteImage.load_from_file(binary_file_path, bands=mask_bands)
    binary_sat_image = SatelliteImage(dataset, array, MASK_BANDS)
    generator = CellGenerator(image=binary_sat_image, size=smallest_window_size)

    result_mask = np.zeros(array.shape, dtype=np.uint8)
    y_matrix = np.zeros(generator.shape)
    for window in generator:
        # for name, feature in iteritems(features.items):
        y = 0
        unique, counts = np.unique(window.raw, return_counts=True)
        # total = np.sum(counts)
        # above_n = np.sum(counts[unique > median])
        # below_n = total - above_n
        # percentage_above = above_n / total
        # if percentage_above > percentage_threshold:
        #      y = 1

        if unique[0] == 0:
            zeros = counts[0]
            non_zeros = np.sum(counts[1:])
            if non_zeros / (zeros + non_zeros) > percentage_threshold:
                y = 255
        else:
            y = 255

        y_matrix[window.x, window.y] = y
        result_mask[window.x_range, window.y_range, 0] = y


    ds, img, bands = load_from_file(image_file, WORLDVIEW3)
    img = normalize_image(img, bands)
    rgb_img = get_rgb_bands(img, bands)

    plt.imshow(rgb_img)

    binary_mask = result_mask
    show_mask = np.ma.masked_where(binary_mask == 0, binary_mask)
    plt.imshow(show_mask[:, :, 0], cmap='jet', interpolation='none', alpha=0.7)
    # plt.title('Binary mask')
    plt.show()
    print('Min {} Max {}'.format(binary_mask.min(), binary_mask.max()))
    print('Len > 0: {}'.format(len(binary_mask[binary_mask > 0])))
    print('Len == 0: {}'.format(len(binary_mask[binary_mask == 0])))

    plt.figure()
    plt.axis('off')
    plt.imshow(result_mask[:, :, 0], cmap='jet', interpolation='none', alpha=1.0)
    plt.show()


    jaccard_index = jaccard_index_binary_masks(truth_mask[:, :, 0], binary_mask[:, :, 0])
    print("Jaccard index: {}".format(jaccard_index))

    y_vector, y_mask = get_y_vector(out_file, smallest_window_size, 0.5, False)
    # assert(np.array_equal(np.reshape(y_vector, y_matrix.shape), y_matrix))

    # y_matrix = np.reshape(y_vector, y_matrix.shape)

    # plt.figure()
    # plt.title("??")
    # plt.imshow(y_matrix, cmap='jet', interpolation='none', alpha=1.0)
    # plt.show()
    # y_mask = np.reshape(y_vector, generator.shape())

    plt.figure()
    plt.axis('off')
    plt.imshow(rgb_img)
    show_mask = np.ma.masked_where(y_mask == 0, y_mask)
    plt.imshow(show_mask[:, :, 0], cmap='jet', interpolation='none', alpha=0.5)
    plt.show()

