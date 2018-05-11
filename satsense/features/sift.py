import numpy as np
import scipy as sp
import cv2
from satsense import SatelliteImage
from .feature import Feature
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.preprocessing import MinMaxScaler
from numba import jit

def sift_cluster(sat_image: SatelliteImage, n_clusters=32, sample_size=100000) -> MiniBatchKMeans:
    sift = cv2.xfeatures2d.SIFT_create()
    kp, descriptors = sift.detectAndCompute(sat_image.gray_ubyte, None)
    del kp  # Free up memory

    # Sample {sample_size} descriptors from all descriptors
    # (Takes random rows) and cluster these
    X = descriptors[np.random.choice(descriptors.shape[0], sample_size, replace=False), :]

    mbkmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42).fit(X)

    return mbkmeans


class Sift(Feature):
    def __init__(self, kmeans: MiniBatchKMeans, windows=((25, 25),)):
        super(Sift, self)
        self.windows = windows
        self.kmeans = kmeans
        self.feature_size = len(self.windows) * kmeans.n_clusters
        self.sift_obj = cv2.xfeatures2d.SIFT_create()

    def __call__(self, cell):
        result = np.zeros(self.feature_size)
        n_clusters = self.kmeans.n_clusters
        for i, window in enumerate(self.windows):
            win = cell.super_cell(window, padding=True)
            start_index = i * n_clusters
            end_index = (i + 1) * n_clusters
            result[start_index: end_index] = self.sift(win.gray_ubyte, self.kmeans)
        return result

    def sift(self, window_gray_ubyte, kmeans: MiniBatchKMeans):
        """
        Calculate the sift feature on the given window

        Args:
            window (nparray): A window of an image
            maximum (int): The maximum value in the image
        """
        kp, descriptors = self.sift_obj.detectAndCompute(window_gray_ubyte, None)
        del kp  # Free up memory



        # Is none if no descriptors are found, i.e. on 0 input range
        if descriptors is None:
            return np.zeros((32))

        codewords = kmeans.predict(descriptors)
        return count_codewords(codewords, 32)


@jit('int32[:](int64[:], int64)', nopython=True)
def count_codewords(codewords, vector_length):
    histogram = np.zeros((vector_length), dtype=np.int32)
    for codeword in codewords:
        histogram[codeword] += 1

    return histogram
