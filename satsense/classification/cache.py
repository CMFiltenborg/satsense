from six import iteritems
from sklearn.externals import joblib
import numpy as np
import os

from satsense.features import FeatureSet


def get_project_root():
    path = os.path.dirname(os.path.realpath(__file__))
    project_root = "{}/../..".format(path)
    return project_root


def cache_model(model, filename):
    dir_path = "{root}/cache/".format(
        root=get_project_root(),
    )

    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)

    full_path = "{dir_path}/{file}.pkl".format(
        dir_path=dir_path,
        file=filename,
    )

    joblib.dump(model, full_path)


def cached_model_exists(filename):
    full_path = "{root}/cache/{file}.pkl".format(
        root=get_project_root(),
        file=filename,
    )

    if os.path.exists(full_path):
        return True

    return False


def load_cached_model(filename: str):
    full_path = "{root}/cache/{file}.pkl".format(
        root=get_project_root(),
        file=filename,
    )

    return joblib.load(full_path)


def cache(array, filename):
    np.save(get_project_root() + "/cache/" + filename + ".npy", array)


def load_cache(filename):
    full_file_path = get_project_root() + "/cache/" + filename + '.npy'

    if not os.path.exists(full_file_path):
        return None

    return np.load(full_file_path)


def load_to_cache(features):
    # Add feature keys
    to_cache = {}
    for name, feature in features.items.items():
        to_cache[name] = {}
    return to_cache


def load_feature_cache(features: FeatureSet, image_name: str, window_size: (int, int)) -> dict:
    cached = {}
    for name, feature in features.items.items():
        feature_cache_key = "{feature}{fwindow}-window{window}-image-{image_name}".format(
            image_name=image_name,
            fwindow=str(feature.windows),
            window=window_size,
            feature=str(feature),
        )
        full_path = "{root}/cache/features/{feature}/{cache_key}.npz".format(
            root=get_project_root(),
            feature=str(feature),
            cache_key=feature_cache_key
        )

        try:
            if os.path.isfile(full_path):
                npzdict = np.load(full_path)
                cached[name] = npzdict
            else:
                cached[name] = {}
        except OSError:
            cached[name] = {}

    return cached


def make_feature_cache_key(feature, image_name, window, window_size):
    return "feature-window.x={x}-window.y={y}-{feature}{fwindow}-window{window}-image-{image_name}".format(
        image_name=image_name,
        fwindow=str(feature.windows),
        window=window_size,
        feature=str(feature),
        x=window.x,
        y=window.y,
    )


def cache_calculated_features(features, image_name, to_cache, window_size):
    for name, feature in features.items.items():
        feature_cache_key = "{feature}{fwindow}-window{window}-image-{image_name}".format(
            image_name=image_name,
            fwindow=str(feature.windows),
            window=window_size,
            feature=str(feature),
        )

        dir_path = "{root}/cache/features/{feature}".format(
            feature=str(feature),
            root=get_project_root(),
        )
        full_path = "{dir_path}/{cache_key}.npz".format(
            dir_path=dir_path,
            cache_key=feature_cache_key,
        )
        if not os.path.isdir(dir_path):
            os.mkdir(dir_path)

        arrays = to_cache[name]
        np.savez(full_path, **arrays)


def cache_file_exists(cache_key):
    return os.path.exists(cache_key)
