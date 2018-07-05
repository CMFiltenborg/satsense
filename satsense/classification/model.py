import itertools
import json
import os
import time

import gdal
import numpy as np
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN

from satsense import SatelliteImage, extract_features
from satsense.bands import MASK_BANDS, WORLDVIEW3
from satsense.classification.classifiers import all_classifiers
from satsense.features.texton import Texton, texton_cluster
from satsense.util.cache import load_cache, cache, cached_model_exists, load_cached_model, cache_model
from satsense.features import FeatureSet
from satsense.features.sift import Sift, sift_cluster
from satsense.generators import CellGenerator


def get_y_vector(binary_file_path: str, smallest_window_size: tuple, percentage_threshold: float = 0.5,
                 cached: bool = False) -> tuple:
    # TODO: Fix cache key.
    if cached:
        y_train = load_cache('y_train')
        if y_train is not None:
            return y_train

    dataset = gdal.Open(binary_file_path, gdal.GA_ReadOnly)
    array = dataset.ReadAsArray()
    array = np.min(array, 0)
    array = array[:, :, np.newaxis]

    binary_sat_image = SatelliteImage(dataset, array, MASK_BANDS)
    generator = CellGenerator(image=binary_sat_image, size=smallest_window_size)

    # Mask which covers the whole image (exploded back up to the real dims)
    real_mask = np.zeros(array.shape, dtype=np.uint8)
    # Y matrix in dims of the blocks
    y_matrix = np.zeros(generator.shape())
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
                y = 1
        else:
            y = 1

        y_matrix[window.x, window.y] = y
        real_mask[window.x_range, window.y_range, 0] = y

    y_train = y_matrix.flatten()
    if cached:
        cache(y_train, "y_train")

    return y_train, real_mask


def get_x_matrix(sat_image: SatelliteImage, image_name, feature_set, window_size=(25, 25), cached=True):
    # image_name = "17FEB16053453-M2AS_R1C2-056239125020_01_P010"
    # image_file = "/home/max/Documents/ai/scriptie/data/%s.TIF" % image_name

    feature_string = feature_set.string_presentation()
    cache_key = "X-{0}-{1}-{2}".format(image_name, str(window_size), feature_string)

    if cached:
        X = load_cache(cache_key)
        if X is not None:
            print("Loaded cached X matrix: {}".format(cache_key))
            return X
    print("X matrix not cached: {}".format(cache_key))

    # bands = WORLDVIEW2
    # sat_image = SatelliteImage.load_from_file(image_file, bands)

    # Calculate PANTEX feature for satellite image
    # Calculates Z features, resulting dimensions is:
    # [M x N x Z], where 0,0,: are the features of the first block
    # In this case we have 1 feature per block

    start = time.time()

    generator = CellGenerator(image=sat_image, size=window_size)
    calculated_features = extract_features(feature_set, generator, load_cached=cached, image_name=image_name)

    end = time.time()
    delta = (end - start)
    print("Calculating multiprocessing im:{} took {} seconds block size: {}, ".format(image_name, delta, window_size))

    if len(calculated_features.shape) == 3:
        nrows = calculated_features.shape[0] * calculated_features.shape[1]
        nfeatures = calculated_features.shape[2]
        X = calculated_features.reshape((nrows, nfeatures))

    # Reshape if we only have one feature, as scikit learn always needs 2 dims
    if len(calculated_features.shape) == 2:
        X = calculated_features.ravel()
        X = X.reshape(-1, 1)

    cache(X, cache_key)
    return X


def balance_dataset(X: np.ndarray, y: np.ndarray, class_ratio=1.3):
    """
    Balances the dataset over the classes
    Class ratio 1.5 means that there are 50% more non-slum examples compared to slum examples

    Zero and one reference the class
    filters part of the X matrix out, where y = 0
    :param X:
    :param y:
    :param class_ratio:
    """
    to_take = round(len(y[y == 1]) * class_ratio)

    X_zeros = X[y == 0, :]
    y_zeros = y[y == 0]

    row_indices = np.random.choice(X_zeros.shape[0], to_take, replace=False)
    X_zeros = X_zeros[row_indices, :]
    y_zeros = y_zeros[row_indices]

    X_ones = X[y == 1, :]
    y_ones = y[y == 1]

    X = np.append(X_zeros, X_ones, axis=0)
    y = np.append(y_zeros, y_ones)

    return X, y


def create_models(images, feature_set: FeatureSet, base_data_path, extension='tif', main_window_size=(30, 30),
                  percentage_threshold=0.5, class_ratio=1.3, bands=WORLDVIEW3, cached=True):
    """
    Yields models, a tuple of (X vector, y vector, real_mask image) for various images
    Also yields a grouped model of all images for classification use

    :param cached:
    :param images:
    :param feature_set:
    :param base_data_path:
    :param extension:
    :param main_window_size:
    :param percentage_threshold:
    :param class_ratio:
    :param bands:
    """
    data = []
    for group_num, image_name in enumerate(images):
        image_file = "{base_path}/{image_name}.{extension}".format(
            base_path=base_data_path,
            image_name=image_name,
            extension=extension
        )
        mask_full_path = "{base_path}/{image_name}_masked.tif".format(base_path=base_data_path, image_name=image_name)
        sat_image = SatelliteImage.load_from_file(image_file, bands)

        X = get_x_matrix(sat_image, image_name=image_name, feature_set=feature_set, window_size=main_window_size,
                         cached=cached)
        y, real_mask = get_y_vector(mask_full_path, main_window_size, percentage_threshold, cached=False)
        # X, y = balance_dataset(X, y, class_ratio=class_ratio)

        print("X shape {}, y shape {}".format(X.shape, y.shape))
        image_vars = (X, y, real_mask, np.full(y.shape, group_num))
        data.append(image_vars)
        # yield image_vars

    if len(data) > 1:
        group_num = 0
        base_X, base_y, _, groups = data[0]
        groups = np.full(base_y.shape, group_num)
        print("X shape {}, y shape {}".format(base_X.shape, base_y.shape))
        for X, y, _, im_groups in data[1:]:
            base_X = np.append(base_X, X, axis=0)
            base_y = np.append(base_y, y, axis=0)
            groups = np.append(groups, im_groups, axis=0)

        return (base_X, base_y, None, groups)


def create_sift_feature(sat_image: SatelliteImage, window_sizes, image_name, n_clusters=32, cached=True) -> Sift:
    cache_key = "kmeans-sift-{}".format(image_name)

    if cached and cached_model_exists(cache_key):
        kmeans = load_cached_model(cache_key)
        print("Loaded cached kmeans {}".format(cache_key))
    else:
        print("Computing k-means model")
        kmeans = sift_cluster(sat_image, n_clusters=n_clusters)
        cache_model(kmeans, cache_key)

    feature = Sift(kmeans, windows=(window_sizes))

    return feature


def create_texton_feature(sat_image: SatelliteImage, window_sizes, image_name, n_clusters=32, cached=True) -> Texton:
    cache_key = "kmeans-texton-{}".format(image_name)

    if cached and cached_model_exists(cache_key):
        kmeans = load_cached_model(cache_key)
        print("Loaded cached kmeans {}".format(cache_key))
    else:
        print("Computing k-means model")
        kmeans = texton_cluster([sat_image], n_clusters=n_clusters)
        cache_model(kmeans, cache_key)

    feature = Texton(kmeans, windows=(window_sizes))

    return feature


def generate_tests(features, classifiers=None):
    if classifiers is None:
        classifiers = all_classifiers()

    for cl in classifiers:
        for fs in generate_feature_sets(features):
            yield (fs, cl)


def generate_classifiers():
    return all_classifiers()


def generate_feature_sets(features):
    """
    Yields all possible combinations of features, of size 1 - len(features)
    :param features:
    """
    combi_sizes = reversed(range(1, len(features) + 1))
    combi_sizes = [len(features), 1]
    for combi_size in combi_sizes:
        combis = itertools.combinations(features, combi_size)
        for combination in combis:
            feature_set = FeatureSet()
            for f in combination:
                feature_set.add(f)

            yield feature_set


def cv_train_test_split_images(images):
    im_set = set(images)
    for test_image in images:
        train_images = im_set.difference({test_image})

        yield (test_image, train_images)


def save_classification_results(results, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as fp:
        json.dump(results, fp, indent=4)


def generate_feature_scales(feature_scales):
    # combi_sizes = reversed(range(1, len(feature_scales) + 1))
    combi_sizes = (1,)

    # !yield all possible scale combinations
    for combi_size in combi_sizes:
        combis = itertools.combinations(feature_scales, combi_size)
        yield from combis


def generate_oversamplers():
    return [
        # ('RandomOversampler', RandomOverSampler(random_state=0)),
        ('SMOTE', SMOTE(n_jobs=-1)),
        # ('ADASYN', ADASYN(n_jobs=-1)),
        (None, None),
    ]
