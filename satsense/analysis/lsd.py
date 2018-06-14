import cv2

import matplotlib.pyplot as plt
from skimage import img_as_ubyte
from skimage.draw import line
from skimage.transform import pyramid_reduce
from skimage.filters import rank
import skimage.morphology as morp
from satsense import SatelliteImage, WORLDVIEW3, extract_features, MONOCHROME
from satsense.features import Feature, FeatureSet
import numpy as np
import scipy as sp
from skimage.feature import canny as canny_edge
import numba
from numba import jit, prange
import time
import gdal

from satsense.features.lacunarity import create_lacunarity
from satsense.generators import CellGenerator
from satsense.image import normalize_image

base_path = "/home/max/Documents/ai/scriptie/data/Clip"
image_name = 'section_3'
extension = 'tif'
image_file = "{base_path}/{image_name}.{extension}".format(
    base_path=base_path,
    image_name=image_name,
    extension=extension
)

# image_file = "/home/max/Documents/ai/scriptie/data/17FEB16053453-M2AS_R1C2-056239125020_01_P010.TIF"

bands = WORLDVIEW3
sat_image = SatelliteImage.load_from_file(image_file, bands)
generator = CellGenerator(image=sat_image, size=(25, 25))

edged_path = "{base_path}/{image_name}_edge.{extension}".format(
    base_path=base_path,
    image_name=image_name,
    extension=extension
)
dataset = gdal.Open(edged_path, gdal.GA_ReadOnly)
edged = dataset.ReadAsArray()
edged = np.expand_dims(edged, axis=2)
edged = normalize_image(edged, MONOCHROME)

scale_size = 2
edged = pyramid_reduce(edged, downscale=scale_size)
edged = img_as_ubyte(edged)

print(edged.shape, np.min(edged), np.max(edged))
plt.figure()
plt.axis('off')
plt.imshow(edged[:, :, 0], cmap='gray')
plt.show()


# create_lacunarity(sat_image)

lsd = cv2.createLineSegmentDetector(0)
detections = lsd.detect(edged)
detected_lines = detections[0]
print(detections)
# detected_lines = [l * 2 for l in detected_lines]
detected_lines *= scale_size
print("Lines found {}".format(len(detected_lines)))
del detections


line_image = np.zeros((sat_image.shape[0], sat_image.shape[1], 1))
mask_image = np.zeros((sat_image.shape[0], sat_image.shape[1], 1))
for dl in detected_lines:
    r0, c0, r1, c1 = dl[0]
    rr, cc = line(int((c0)), int((r0)), int((c1)), int((r1)))
    line_image[rr, cc] = 1


plt.figure()
plt.axis('off')
plt.title('LSD')
plt.imshow(line_image[:, :, 0], cmap='gray')
plt.show()


# mask_image = lsd.drawSegments(mask_image, detected_lines)
gray_ubyte = sat_image.gray_ubyte
gray_ubyte = lsd.drawSegments(gray_ubyte, detected_lines)


plt.figure()
plt.axis('off')
plt.title('LSD')
plt.imshow(gray_ubyte, cmap='gray')
plt.show()



step_size = 100
for i in range(0, mask_image.shape[0], step_size):
    for j in range(0, mask_image.shape[1], step_size):
        cnt = np.sum(line_image[i:i + step_size, j:j + step_size])
        if cnt >= 2:
            mask_image[i:i + step_size, j:j + step_size] = True

plt.figure()
plt.axis('off')
plt.title('LSD Mask')
plt.imshow(mask_image[:, :, 0], cmap='gray')

plt.show()


def create_lacunarity(sat_image: SatelliteImage):
    lsd = cv2.createLineSegmentDetector(0)

    sigma = 0.5
    # raw = np.copy(sat_image.raw)
    normalized = np.copy(sat_image.normalized)
    scale_size = 3
    normalized = pyramid_reduce(normalized, downscale=scale_size)
    rgb = get_rgb_bands(normalized, sat_image.bands)
    grayscale = get_grayscale_image(rgb, RGB)
    img_ubyte = img_as_ubyte(grayscale)
    # img_ubyte = grayscale

    print(sat_image.shape)
    print(img_ubyte.shape)
    print(img_ubyte.dtype)
    print(np.min(img_ubyte), np.max(img_ubyte))


    # local histogram equalization
    img_ubyte = rank.equalize(img_ubyte, selem=morp.disk(100))


    # compute the median of the single channel pixel intensities
    v = np.median(img_ubyte)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(img_ubyte, lower, upper)

    # edged = pyramid_expand(edged, upscale=2)
    print(edged.dtype)
    print(edged.shape)


    # edged = sat_image.canny_edged
    plt.figure()
    plt.imshow(edged, cmap='gray')
    plt.show()

    print(sat_image.canny_edged.dtype)
    # edged = sat_image.canny_edged.astype(np.uint8)
    edged *= 255
    edged = edged.astype(np.uint8)
    edged *= 255

    print(edged.dtype)
    print(edged.shape)
    print(np.min(edged), np.max(edged))
    # plt.figure()
    # plt.imshow(edged, cmap='gray')
    # plt.show()

    detections = lsd.detect(edged)
    detected_lines = detections[0]
    print(detections)
    # detected_lines = [l * 2 for l in detected_lines]
    detected_lines *= scale_size
    print("Lines found {}".format(len(detected_lines)))
    del detections


    mask_image = np.zeros((sat_image.shape[0], sat_image.shape[1], 1))
    gray_ubyte = sat_image.gray_ubyte
    gray_ubyte = lsd.drawSegments(gray_ubyte, detected_lines)


    plt.figure()
    plt.imshow(gray_ubyte, cmap='gray')

    plt.show()


    # edged = sat_image.canny_edged
    # generator = CellGenerator(sat_image, (100, 100))

    step_size = 100
    for i in range(0, mask_image.shape[0], step_size):
        for j in range(0, mask_image.shape[1], step_size):
            cnt = np.sum(mask_image)
            if cnt >= 2:
                mask_image[i:i + step_size, j:j + step_size] = True

    plt.figure()
    plt.imshow(mask_image[:, :, 0], cmap='gray')

    plt.show()


    return mask_image