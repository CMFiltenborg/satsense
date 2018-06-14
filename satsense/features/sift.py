import numpy as np
import scipy as sp
import cv2

from satsense.generators import CellGenerator
from satsense import SatelliteImage
from satsense.generators.cell_generator import super_cell
from .feature import Feature
from sklearn.cluster import MiniBatchKMeans
from typing import Iterator

def sift_cluster(sat_images: Iterator[SatelliteImage], n_clusters=32, sample_size=100000) -> MiniBatchKMeans:
    sift = cv2.xfeatures2d.SIFT_create()
    base_descriptors = None
    for sat_image in sat_images:
        kp, descriptors = sift.detectAndCompute(sat_image.gray_ubyte, None)
        del kp  # Free up memory

        # Add descriptors if we already had some
        if base_descriptors is None:
            base_descriptors = descriptors
        else:
            base_descriptors = np.append(base_descriptors, descriptors, axis=0)


    # Sample {sample_size} descriptors from all descriptors
    # (Takes random rows) and cluster these
    X = base_descriptors[np.random.choice(base_descriptors.shape[0], sample_size, replace=False), :]

    mbkmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42).fit(X)

    return mbkmeans


def sift_for_chunk(chunk, kmeans, normalized=True):
    chunk_len = len(chunk)
    cluster_count = kmeans.n_clusters
    sift_obj = cv2.xfeatures2d.SIFT_create()

    coords = np.zeros((chunk_len, 2))
    chunk_matrix = np.zeros((chunk_len, cluster_count), dtype=np.float64)
    for i in range(chunk_len):
        coords[i, :] = chunk[i][0:2]

        win_gray_ubyte = chunk[i][2]

        kp, descriptors = sift_obj.detectAndCompute(win_gray_ubyte, None)
        del kp  # Free up memory

        if descriptors is None:
            chunk_matrix[i, :] = np.zeros((cluster_count), dtype=np.int32)
            continue

        codewords = kmeans.predict(descriptors)
        counts = np.bincount(codewords, minlength=cluster_count)

        # Perform normalization
        if normalized:
            counts = counts / cluster_count

        chunk_matrix[i, :] = counts

    return coords, chunk_matrix


class Sift(Feature):
    def __init__(self, kmeans: MiniBatchKMeans, windows=((25, 25),), normalized=True):
        super(Sift, self)
        self.windows = windows
        self.kmeans = kmeans
        self.feature_size = len(self.windows) * kmeans.n_clusters
        self.sift_obj = cv2.xfeatures2d.SIFT_create()
        self.normalized = normalized

    def __call__(self, chunk):
        return sift_for_chunk(chunk, self.kmeans, self.normalized)

        # result = np.zeros(self.feature_size)
        # n_clusters = self.kmeans.n_clusters
        # for i, window in enumerate(self.windows):
        #     win = cell.super_cell(window, padding=True)
        #     start_index = i * n_clusters
        #     end_index = (i + 1) * n_clusters
        #     result[start_index: end_index] = self.sift(win.gray_ubyte, self.kmeans)
        # return result

    def __str__(self):
        normalized = "n" if self.normalized == True else "nn"
        return "Si-{}{}".format(str(self.windows), normalized)


    def initialize(self, generator: CellGenerator, scale):
        data = []
        for window in generator:
            win_gray_ubyte, _, _ = super_cell(generator.image.gray_ubyte, scale, window.x_range, window.y_range, padding=False)
            processing_tuple = (window.x, window.y, win_gray_ubyte)
            data.append(processing_tuple)

        return data


    def sift(self, window_gray_ubyte, kmeans: MiniBatchKMeans):
        """
        Calculate the sift feature on the given window
        """
        kp, descriptors = self.sift_obj.detectAndCompute(window_gray_ubyte, None)
        del kp  # Free up memory


        # Is none if no descriptors are found, i.e. on 0 input range
        cluster_count = kmeans.n_clusters
        if descriptors is None:
            return np.zeros((cluster_count), dtype=np.int32)

        codewords = kmeans.predict(descriptors)
        counts = np.bincount(codewords, minlength=cluster_count)

        # Perform normalization
        if self.normalized:
            counts = counts / cluster_count

        return counts
