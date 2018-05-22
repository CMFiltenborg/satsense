from satsense.util.cache import load_to_cache, load_feature_cache, make_feature_cache_key, \
    cache_calculated_features
from .features import FeatureSet
from .generators import CellGenerator
import numpy as np
from six import iteritems
import sys


def extract_features(features: FeatureSet, generator: CellGenerator, load_cached=True, image_name=""):
    shape = generator.shape()
    window_size = (generator.x_size, generator.y_size)

    total_length = features.index_size
    print("Total length found: ", total_length)
    feature_vector = np.zeros((shape[0], shape[1], total_length))
    print("Feature vector:")
    print(feature_vector.shape)

    cache_hits = 0
    cache_misses = 0
    windows_processed = 0
    total_windows = generator.x_length * generator.y_length

    if load_cached:
        cached = load_feature_cache(features, image_name, window_size)
    to_cache = load_to_cache(features)

    for name, feature in iteritems(features.items):
        if hasattr(feature, 'initialize'):
            feature.initialize(generator.image)

    for window in generator:
        for name, feature in iteritems(features.items):
            feature_cache_key = make_feature_cache_key(feature, image_name, window, window_size)

            # If we have this feature cached we just load it from there
            # Has to be cached at exact same windows (= same dims)
            if load_cached and feature_cache_key in cached[name]:
                cache_hits += 1
                feature_array = cached[name][feature_cache_key]
                feature_vector[window.x, window.y, feature.indices] = feature_array

                continue

            # Calculate feature and set in feature_vector
            feature_array = feature(window)
            feature_vector[window.x, window.y, feature.indices] = feature_array

            cache_misses += 1
            # Save calculations so we can cache these later
            to_cache[name][feature_cache_key] = feature_array

        # Re-draws a progress bars every 500 windows
        windows_processed += 1
        percentage = 1 / total_windows * windows_processed
        if percentage > 0 and windows_processed % 500 == 0:
            draw_progress_bar(percentage)

    cache_calculated_features(features, image_name, to_cache, window_size)

    print("\n")
    print("Had {} cache hits, {} misses".format(cache_hits, cache_misses))

    return feature_vector


def draw_progress_bar(percent, barLen=20):
    sys.stdout.write("\r")
    progress = ""
    for i in range(barLen):
        if i < int(barLen * percent):
            progress += "="
        else:
            progress += " "
    sys.stdout.write("\r[ %s ] %.2f%%" % (progress, percent * 100))
    sys.stdout.flush()
