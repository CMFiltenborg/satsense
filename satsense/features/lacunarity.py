import numpy as np
from numba import jit, prange

from satsense.generators import CellGenerator
from satsense.generators.cell_generator import super_cell
from satsense.image import SatelliteImage
from satsense.features import Feature


# @jit("float64(boolean[:, :], int64)", nopython=True, parallel=True)
@jit("float64(boolean[:, :], int64)", nopython=True)
def lacunarity(edged_image, box_size):
    """
    Calculate the lacunarity value over an image, following these papers:

    Kit, Oleksandr, and Matthias Lüdeke. "Automated detection of slum area change in Hyderabad, India using multitemporal satellite imagery." ISPRS journal of photogrammetry and remote sensing 83 (2013): 130-137.

    Kit, Oleksandr, Matthias Lüdeke, and Diana Reckien. "Texture-based identification of urban slums in Hyderabad, India using remote sensing data." Applied Geography 32.2 (2012): 660-667.
    """

    # accumulator holds the amount of ones for each position in the image, defined by a sliding window
    accumulator = np.zeros(edged_image.shape)
    for i in prange(edged_image.shape[0] - (box_size)):
        for j in prange(edged_image.shape[1] - (box_size)):
            # sum the binary-box for the amount of 1s in this box
            # box = edged_image[j:j + box_size, j: j + box_size]
            accumulator[i, j] = np.sum(edged_image[j:j + box_size, i: i + box_size])

    accumulator = accumulator.flatten()
    mean_sqrd = np.mean(accumulator) ** 2
    if mean_sqrd == 0:
        return 0.0

    return (np.var(accumulator) / mean_sqrd) + 1


@jit
def lacunarity_for_chunk(chunk, scales, box_sizes):
    len_box_sizes = len(box_sizes)
    chunk_len = len(chunk)

    scales_len = len(scales)
    chunk_matrix = np.zeros((chunk_len, scales_len * len_box_sizes))
    coords = np.zeros((chunk_len, 2))
    for i in range(0, chunk_len, scales_len):
        # Set coordinates
        coords[i, :] = chunk[i][0:2]

        # feature_vector = np.zeros(len(scales) * len(box_sizes))
        for k in range(scales_len):
            x, y, scale, edged = chunk[i + k]
            for j in range(len_box_sizes):
                box_size = box_sizes[j]
                chunk_matrix[i, k + j] = lacunarity(edged, box_size)

    return (coords, chunk_matrix)



class Lacunarity(Feature):
    def __init__(self, windows=((25, 25),), box_sizes=(10, 20, 30)):
        super(Lacunarity, self)
        self.box_sizes = box_sizes
        self.windows = windows
        self.feature_size = len(self.windows) * len(box_sizes)

    def __call__(self, cell):
        # dt = np.dtype({'names':['x', 'y', 'fv'], 'formats':[np.int64, np.int64, '({},)float64'.format(len(self.windows) * len(self.box_sizes))]})
        return lacunarity_for_chunk(cell, self.windows, self.box_sizes)
        # result = np.zeros(self.feature_size)
        # len_box_sizes = len(self.box_sizes)
        # for i, scale in enumerate(self.windows):
        #     win = cell.super_cell(scale, padding=True)
        #     edged = win.canny_edged
        #
        #     # For every box size we have a feature for this scale
        #     for j in range(len_box_sizes):
        #         box_size = self.box_sizes[j]
        #         result[i + j] = lacunarity(edged, box_size)
        #
        # return result

    def initialize(self, generator: CellGenerator):
        # Load the canny edged image for the whole image
        # This so it is not done on a window by window basis...
        sat_image = generator.image
        norm = sat_image.normalized
        ce = sat_image.canny_edged

        data = []
        for window in generator:
            for scale in self.windows:
                edged, _, _ = super_cell(generator.image.canny_edged, scale, window.x_range, window.y_range, padding=True)
                processing_tuple = (window.x, window.y, scale, edged)
                data.append(processing_tuple)

        return data

    def __str__(self):
        return "La-{}-{}".format(str(self.windows), str(self.box_sizes))
