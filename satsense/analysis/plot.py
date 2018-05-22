from os import walk
from satsense.bands import *
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import gdal
import pylab

def remap(x, o_min, o_max, n_min, n_max):
    # range check
    if o_min == o_max:
        # print("Warning: Zero input range")
        return None

    if n_min == n_max:
        # print("Warning: Zero output range")
        return None

    # check reversed input range
    reverse_input = False
    old_min = min(o_min, o_max)
    old_max = max(o_min, o_max)
    if not old_min == o_min:
        reverse_input = True

    # check reversed output range
    reverse_output = False
    new_min = min(n_min, n_max)
    new_max = max(n_min, n_max)
    if not new_min == n_min:
        reverse_output = True

    # print("Remapping from range [{0}-{1}] to [{2}-{3}]".format(old_min, old_max, new_min, new_max))
    portion = (x - old_min) * (new_max - new_min) / (old_max - old_min)
    if reverse_input:
        portion = (old_max - x) * (new_max - new_min) / (old_max - old_min)

    result = portion + new_min
    if reverse_output:
        result = new_max - portion

    return result


def normalize_image(image, bands, technique='cumulative',
                    percentiles=[2.0, 98.0], numstds=2):
    """
    Normalizes the image based on the band maximum
    """
    normalized_image = image.copy()
    for name, band in bands.items():
        # print("Normalizing band number: {0} {1}".format(band, name))
        if technique == 'cumulative':
            percents = np.percentile(image[:, :, band], percentiles)
            new_min = percents[0]
            new_max = percents[1]
        elif technique == 'meanstd':
            mean = normalized_image[:, :, band].mean()
            std = normalized_image[:, :, band].std()

            new_min = mean - (numstds * std)
            new_max = mean + (numstds * std)
        else:
            new_min = normalized_image[:, :, band].min()
            new_max = normalized_image[:, :, band].max()

        if new_min:
            normalized_image[normalized_image[:, :, band] < new_min, band] = new_min
        if new_max:
            normalized_image[normalized_image[:, :, band] > new_max, band] = new_max

        normalized_image[:, :, band] = remap(normalized_image[:, :, band], new_min, new_max, 0, 1)

    return normalized_image

def read_geotiff(filename):
    """Yield the bands stored in a geotiff file as numpy arrays."""
    dataset = gdal.Open(filename)
    # read everything
    #     a = dataset.ReadAsArray()
    #     yield a
    # read bands separately
    for band in range(dataset.RasterCount):
        yield dataset.GetRasterBand(band + 1).ReadAsArray()


def img_show(img):
    for band in img:
        print(band.shape, band.dtype, band.min(), band.max())
        pylab.imshow(band)
        pylab.show()


def load_from_file(path, bands):
    """
    Loads the specified path from file and loads the bands into a numpy array
    @returns dataset The raw gdal dataset
            image The image loaded as a numpy array
    """
    dataset = gdal.Open(path, gdal.GA_ReadOnly)
    array = dataset.ReadAsArray()

    if len(array.shape) == 3:
        # The bands column is in the first position, but we want it last
        array = np.rollaxis(array, 0, 3)
    elif len(array.shape) == 2:
        # This image seems to have one band, so we add an axis for ease
        # of use in the rest of the library
        array = array[:, :, np.newaxis]

    image = array.astype('float32')

    return dataset, image, bands


def get_rgb_bands(image, bands):
    """
    Converts the image to rgb format.
    """
    if bands is not MONOCHROME:
        red = image[:, :, bands['red']]
        green = image[:, :, bands['green']]
        blue = image[:, :, bands['blue']]

        img = np.rollaxis(np.array([red, green, blue]), 0, 3)
    else:
        pass
        # img = color.grey2rgb(image)

    return img

def show_worldview2():
    ds, img, bands = load_from_file(source_img, WORLDVIEW2)
    img = normalize_image(img, bands)
    rgb_img = get_rgb_bands(img, bands)

    pylab.imshow(rgb_img)
    pylab.show()

def plot_kde_all(images):
    f, axes = plt.subplots(len(images), sharex=True)
    for i, img in enumerate(images):
        for j, band in enumerate(img):
            print(band.shape, band.dtype, band.min(), band.max())
            sns.set_style('whitegrid')
            sns.kdeplot(band.flatten(), bw=0.5, ax=axes[i])


def plot_kde(img):
    # f, axes = plt.subplots(9)
    for i, band in enumerate(img):
        # print(band.shape, band.dtype, band.min(), band.max())
        sns.set_style('whitegrid')
        sns.kdeplot(band.flatten(), bw=0.5)


def calculate_stats(img):
    series = []
    for band in img:
        stats = {}
        stats['mean'] = np.mean(band)
        stats['max'] = band.max()
        stats['min'] = band.min()

        series.append(pd.Series(stats))

    df = pd.DataFrame(series)

    return df

source_img = "/home/max/Documents/ai/scriptie/spfeas/17FEB16053453-M2AS_R1C2-056239125020_01_P010.TIF"
pattern = "/home/max/Documents/ai/scriptie/spfeas/17FEB16053453-M2AS_R1C2-056239125020_01_P010_features/17FEB16053453-M2AS_R1C2-056239125020_01_P010_features/17FEB16053453-M2AS_R1C2-056239125020_01_P010__BD5-3-2_BK8_SC8-16-32_TRpantex/*.tif"


# pattern = "/home/max/Documents/ai/scriptie/spfeas/17FEB16053453-M2AS_R1C2-056239125020_01_P010_features/17FEB16053453-M2AS_R1C2-056239125020_01_P010_features/17FEB16053453-M2AS_R1C2-056239125020_01_P010__BD5-3-2_BK8_SC8-16-32_TRpantex/17FEB16053453-M2AS_R1C2-056239125020_01_P010__BD5-3-2_BK8_SC8-16-32__ST1-009__TL000001.tif"


# images = [list(read_geotiff(filepath)) for filepath in glob.glob(pattern)]
# images = images[:5]
# # for file in glob.glob(pattern):
# #     data = map(read_geotiff, file)
# #     plot_kde(data)
#
# series = []
# for img in images:
#     plot_kde(img)
#     df = calculate_stats(img)
#     series.append(df.mean())
#
# total_img = np.array([0])
# for img in images:
#     for band in img:
#         total_img = np.append(total_img, band.flatten())
#
# sns.set_style('whitegrid')
# sns.kdeplot(total_img, bw=0.5)
# print(total_img)
#
# total_df = pd.DataFrame(series)
# plot_kde_all(images)
# plot_kde(np.asarray([total_df['mean']]))
#
# plt.show()
# print([b.shape() for b in bands])

# img_show(read_geotiff(source_img))

if __name__ == '__main__':
    show_worldview2()

# ds = gdal.Open(source_img).ReadAsArray()
# print(ds.shape)
# pylab.imshow(ds)
# pylab.show()

# im = plt.imshow(ds)


