import math

import numpy as np
from skimage.filters import gabor_kernel, gaussian
from typing import List, Iterator
from satsense import SatelliteImage, WORLDVIEW3
from satsense.generators import CellGenerator
from satsense.generators.cell_generator import Cell, super_cell
from .feature import Feature
from sklearn.cluster import MiniBatchKMeans
from scipy import ndimage as ndi
from functools import lru_cache


def texton_cluster(sat_images: Iterator[SatelliteImage], n_clusters=32, sample_size=100000) -> MiniBatchKMeans:
    base_descriptors = None

    # prepare filter bank kernels
    kernels = create_kernels()
    for sat_image in sat_images:
        print("Computing texton descriptors for sat_image {}".format(str(sat_image.shape)))
        descriptors = compute_feats(sat_image.grayscale, kernels)
        descriptors = descriptors.reshape((descriptors.shape[0] * descriptors.shape[1], descriptors.shape[2]))

        # Compute Difference of Gaussian
        dog = np.expand_dims((gaussian(sat_image.grayscale, sigma=1) - gaussian(sat_image.grayscale, sigma=3)).ravel(),
                             axis=1)
        descriptors = np.append(descriptors, dog, axis=1)

        # Add descriptors if we already had some
        if base_descriptors is None:
            base_descriptors = descriptors
        else:
            base_descriptors = np.append(base_descriptors, descriptors, axis=0)

    # Sample {sample_size} descriptors from all descriptors
    # (Takes random rows) and cluster these
    print("Sampling from {}".format(str(base_descriptors.shape)))
    sample_size = int(base_descriptors.shape[0] / 2)
    X = base_descriptors[np.random.choice(base_descriptors.shape[0], sample_size, replace=False), :]

    mbkmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42).fit(X)

    return mbkmeans


@lru_cache(maxsize=1)
def create_kernels():
    kernels = []
    angles = 8
    thetas = np.linspace(0, np.pi, angles)
    for theta in thetas:
        # theta = theta / 8. * np.pi
        for sigma in (1,):
            for frequency in (0.05, 0.25):
                kernel = np.real(gabor_kernel(frequency, theta=theta,
                                              sigma_x=sigma, sigma_y=sigma))
                kernels.append(kernel)

    return kernels


def compute_feats(image, kernels):
    feats = np.zeros(image.shape + (len(kernels),), dtype=np.double)
    for k, kernel in enumerate(kernels):
        filtered = ndi.convolve(image, kernel, mode='wrap')
        feats[:, :, k] = filtered

    return feats


def texton_for_chunk(chunk, kmeans, normalized=True):
    chunk_len = len(chunk)
    cluster_count = kmeans.n_clusters

    coords = np.zeros((chunk_len, 2))
    chunk_matrix = np.zeros((chunk_len, cluster_count))
    for i in range(chunk_len):
        coords[i, :] = chunk[i][0:2]

        # sub_descriptors = chunk[i][2]
        # sub_descriptors = sub_descriptors.reshape((sub_descriptors.shape[0] * sub_descriptors.shape[1], sub_descriptors.shape[2]))

        im_grayscale = chunk[i][2]
        sub_descriptors = compute_feats(im_grayscale, create_kernels())

        # Calculate Difference-of-Gaussian
        dog = np.expand_dims(gaussian(im_grayscale, sigma=1) - gaussian(im_grayscale, sigma=3), axis=2)
        sub_descriptors = np.append(sub_descriptors, dog, axis=2)
        sub_descriptors = sub_descriptors.reshape(
            (sub_descriptors.shape[0] * sub_descriptors.shape[1], sub_descriptors.shape[2]))

        codewords = kmeans.predict(sub_descriptors)
        counts = np.bincount(codewords, minlength=cluster_count)

        # Perform normalization
        if normalized:
            counts = counts / cluster_count

        chunk_matrix[i, :] = counts

    return coords, chunk_matrix


class Texton(Feature):
    def __init__(self, kmeans: MiniBatchKMeans, windows=((25, 25),), normalized=True):
        super(Texton, self)
        self.windows = windows
        self.kmeans = kmeans
        self.feature_size = len(self.windows) * kmeans.n_clusters
        self.normalized = normalized
        # self.kernels = create_kernels()
        self.descriptors = None

    def __call__(self, chunk):
        return texton_for_chunk(chunk, self.kmeans, self.normalized)

        # result = np.zeros(self.feature_size)
        # n_clusters = self.kmeans.n_clusters
        # for i, window in enumerate(self.windows):
        #     win = cell.super_cell(window, padding=True)
        #     start_index = i * n_clusters
        #     end_index = (i + 1) * n_clusters
        #     result[start_index: end_index] = self.texton(win, self.kmeans)
        # return result

    def __str__(self):
        normalized = "n" if self.normalized == True else "nn"
        return "Te-{}-{}".format(str(self.windows), normalized)

    def chunk_size(self, cpu_cnt, im_shape):
        # return int(math.ceil(im_shape[0] * im_shape[1] / cpu_cnt))
        return im_shape[0]

    def initialize(self, generator: CellGenerator, scale):
        im_grayscale = generator.image.grayscale

        data = []
        for window in generator:
            im, x_range, y_range = super_cell(im_grayscale, scale, window.x_range, window.y_range, padding=True)
            processing_tuple = (window.x, window.y, im)
            data.append(processing_tuple)

        return data



def test_texton():
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

    texton_cluster(sat_image)
