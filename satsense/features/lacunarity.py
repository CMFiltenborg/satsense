from satsense import SatelliteImage, WORLDVIEW3, extract_features
from satsense.features import Feature, FeatureSet
import numpy as np
import scipy as sp
from skimage.feature import canny as canny_edge
import numba
from numba import jit, prange
import time

from satsense.generators import CellGenerator


def create_lacunarity(sat_image: SatelliteImage, image_name, windows=((25, 25)), cached=True):
    edged = sat_image.canny_edged

    # t0 = time.time()
    # mean, variance = calculate_lacunarity_stats_nb(edged, 2)
    # t1 = time.time()
    # total = t1 - t0
    # print("Numba version took {}".format(total))
    #
    # t0 = time.time()
    # mean, variance = calculate_lacunarity_stats(edged, 2)
    # t1 = time.time()
    # total = t1 - t0
    # print("Non-numba version took {}".format(total))

    return Lacunarity(windows=windows)


def lacunarity_python(edged_image, box_size):
    # accumulator holds the amount of ones for each position in the image, defined by a sliding window
    #
    accumulator = np.zeros(edged_image.shape)
    for i in range(edged_image.shape[0] - box_size):
        for j in range(edged_image.shape[1] - box_size):
            # sum the binary-box for the amount of 1s in this box
            # box = edged_image[j:j + box_size, j: j + box_size]
            accumulator[i, j] = np.sum(edged_image[j:j + box_size, i: i + box_size])

    accumulator = accumulator.flatten()
    mean = np.mean(accumulator)
    if mean == 0:
        return 0.0

    return np.var(accumulator) / mean ** 2 + 1


@jit("float64(boolean[:, :], int64)", nopython=True, parallel=True)
def lacunarity(edged_image, box_size):
    # accumulator holds the amount of ones for each position in the image, defined by a sliding window
    #
    accumulator = np.zeros(edged_image.shape)
    for i in prange(edged_image.shape[0] - (box_size)):
        for j in prange(edged_image.shape[1] - (box_size)):
            # sum the binary-box for the amount of 1s in this box
            # box = edged_image[j:j + box_size, j: j + box_size]
            accumulator[i, j] = np.sum(edged_image[j:j + box_size, i: i + box_size])

    accumulator = accumulator.flatten()
    mean = np.mean(accumulator)
    if mean == 0:
        return 0.0

    return np.var(accumulator) / mean ** 2 + 1


class Lacunarity(Feature):
    def __init__(self, windows=((25, 25),), box_sizes=(2, 4, 6)):
        super(Lacunarity, self)
        self.box_sizes = box_sizes
        self.windows = windows
        self.feature_size = len(self.windows) * len(box_sizes)

    @jit
    def __call__(self, cell):
        result = np.zeros(self.feature_size)
        len_box_sizes = len(self.box_sizes)
        for i, window in enumerate(self.windows):
            win = cell.super_cell(window, padding=True)

            # For every box size we have a feature for this window
            for j in range(len_box_sizes):
                box_size = self.box_sizes[j]
                result[i + j] = lacunarity(win.canny_edged, box_size)

        return result


def test_numba_performance(benchmark):
    print("????")
    base_path = "/home/max/Documents/ai/scriptie/data/Clip"
    image_name = 'section_1'
    extension = 'tif'
    image_file = "{base_path}/{image_name}.{extension}".format(
        base_path=base_path,
        image_name=image_name,
        extension=extension
    )
    bands = WORLDVIEW3
    sat_image = SatelliteImage.load_from_file(image_file, bands)
    generator = CellGenerator(image=sat_image, size=(25, 25))

    edged = sat_image.canny_edged
    # benchmark(lacunarity, *(edged, 2))
    benchmark(run_lacunarity, generator)


def run_lacunarity(generator):
    lacunarity = Lacunarity(windows=((100, 100),), box_sizes=(2, 4, 6))
    feature_set = FeatureSet()
    feature_set.add(lacunarity, name="LACUNARITY")
    extract_features(feature_set, generator, False, 'section_1')

    # for window in generator:
    #     feature_array = lacunarity(window)

def test_python_performance(benchmark):
    print("????")
    base_path = "/home/max/Documents/ai/scriptie/data/Clip"
    image_name = 'section_1'
    extension = 'tif'
    image_file = "{base_path}/{image_name}.{extension}".format(
        base_path=base_path,
        image_name=image_name,
        extension=extension
    )
    bands = WORLDVIEW3
    sat_image = SatelliteImage.load_from_file(image_file, bands)

    edged = sat_image.canny_edged
    benchmark(lacunarity_python, *(edged, 2))

