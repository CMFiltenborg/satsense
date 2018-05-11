import gdal
import numpy as np

from satsense import SatelliteImage, extract_features
from satsense.bands import MASK_BANDS, WORLDVIEW3
from satsense.classification.cache import load_cache, cache
from satsense.features import FeatureSet
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
    cache(y_train, "y_train")

    return y_train, real_mask


def get_x_matrix(sat_image: SatelliteImage, image_name, feature_set, window_size=(25, 25), cached=True):
    # image_name = "17FEB16053453-M2AS_R1C2-056239125020_01_P010"
    # image_file = "/home/max/Documents/ai/scriptie/data/%s.TIF" % image_name

    feature_string = feature_set.string_presentation()
    cache_key = "x_train-{0}-{1}-{2}".format(image_name, str(window_size), feature_string)

    if cached:
        X = load_cache(cache_key)
        if X is not None:
            print("Loaded cached X matrix: {}".format(cache_key))
            return X
    print("X matrix not cached: {}".format(cache_key))

    # bands = WORLDVIEW2
    # sat_image = SatelliteImage.load_from_file(image_file, bands)

    generator = CellGenerator(image=sat_image, size=window_size)

    # Calculate PANTEX feature for satellite image
    # Calculates Z features, resulting dimensions is:
    # [M x N x Z], where 0,0,: are the features of the first block
    # In this case we have 1 feature per block
    calculated_features = extract_features(feature_set, generator, image_name=image_name)

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


def create_feature_set(features, sat_image: SatelliteImage) -> FeatureSet:
    pass


def create_models(images, features, base_data_path, extension='tif', main_window_size=(30, 30), percentage_threshold=0.5, class_ratio=1.3, bands=WORLDVIEW3):

    data = []
    for image_name in images:
        image_file = "{base_path}/{image_name}.{extension}".format(
            base_path=base_data_path,
            image_name=image_name,
            extension=extension
        )
        bands = WORLDVIEW3
        mask_full_path = "{base_path}/{image_name}_masked.tif".format(base_path=base_data_path, image_name=image_name)
        sat_image = SatelliteImage.load_from_file(image_file, bands)

        feature_set = create_feature_set(features, sat_image)

        X_image = get_x_matrix(sat_image, image_name=image_name, feature_set=feature_set, window_size=main_window_size,
                         cached=True)
        y_image, real_mask = get_y_vector(mask_full_path, main_window_size, percentage_threshold, cached=False)
        X, y = balance_dataset(X_image, y_image, class_ratio=class_ratio)

        image_vars = (X_image, y_image, real_mask)
        data.append(image_vars)
        yield image_vars

    if len(data) > 1:
        group_num = 0
        base_X, base_y, _ = data[0]
        groups = np.full(base_y.shape, group_num)
        for X, y, _ in data[0+1:]:
            group_num += 1

            base_X = np.append(base_X, X)
            base_y = np.append(base_y, y)
            groups = np.append(groups, np.full(base_y.shape, group_num))

        yield (base_X, base_y, None)



