import math

from numba import jit, njit

from satsense.bands import WORLDVIEW3
from satsense.generators.cell_generator import super_cell
from satsense.image import SatelliteImage
from satsense.features.lacunarity import Lacunarity, lacunarity, lacunarity_for_chunk
from satsense.util.cache import load_to_cache, load_feature_cache, make_feature_cache_key, \
    cache_calculated_features, load_cache, cache
from satsense.features import FeatureSet
from satsense.generators import CellGenerator
import numpy as np
from six import iteritems
import sys
import dask.array as da
from dask import delayed
import dask
import dask.bag as db
# from dask.diagnosics import ProgressBar
import time

from multiprocessing import cpu_count
from multiprocessing import Pool #  Process pool
from multiprocessing import sharedctypes
from functools import partial

import os
import resource



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
        # if windows_processed == 450:
        #     break
        percentage = 1 / total_windows * windows_processed
        if percentage > 0 and windows_processed % 500 == 0:
            draw_progress_bar(percentage)

    # cache_calculated_features(features, image_name, to_cache, window_size)

    print("\n")
    print("Had {} cache hits, {} misses".format(cache_hits, cache_misses))

    return feature_vector

def extract_features_conc(features: FeatureSet, generator: CellGenerator, load_cached=True, image_name=""):
    shape = generator.shape()
    window_size = (generator.x_size, generator.y_size)

    total_length = features.index_size
    print("Total length found: ", total_length)
    feature_vector = np.zeros((shape[0], shape[1], total_length))
    # feature_vector = da.from_array(feature_vector, chunks=(100, 100, total_length))
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

    feature_vectors = []
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
            feature_array = delayed(feature)(window)
            # feature_vector[window.x, window.y, feature.indices] = feature_array
            # feature_vectors.append(delayed(feature_array))
            r = ([window.x, window.y, feature.indices, feature_array])
            feature_vectors.append(r)

            cache_misses += 1
            # Save calculations so we can cache these later
            to_cache[name][feature_cache_key] = feature_array

        # Re-draws a progress bars every 500 windows
        windows_processed += 1
        # if windows_processed == 450:
        #     break
        percentage = 1 / total_windows * windows_processed
        if percentage > 0 and windows_processed % 500 == 0:
            draw_progress_bar(percentage)


    # partition in 100-sized chunks
    feature_vectors = [delayed(feature_vectors[i:i+100]) for i in range(0, len(feature_vectors), 100)]

    bag = db.from_delayed(feature_vectors)
    print("Visualizing")
    # bag.visualize(filename='graph')
    print("done visualizing")
    with ProgressBar():
        feature_vectors = list(bag)
    # print("I do not come here GVD")
    # print(feature_vectors)
    for row in feature_vectors:
        x, y, feature_indices, feature_array = row
        # print(x, y, feature_array)
        feature_vector[x, y, feature_indices] = feature_array


    # cache_calculated_features(features, image_name, to_cache, window_size)

    print("\n")
    print("Had {} cache hits, {} misses".format(cache_hits, cache_misses))

    return feature_vector

# def compute_to_array(x, feature_vector):
#     window, feature, feature_array = x
#     feature_vector[window.x, window.y, feature.indices] = feature_array


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



def extract_features_futur(features: FeatureSet, generator: CellGenerator, load_cached=True, image_name=""):
    start = time.time()
    shape = generator.shape()
    window_size = (generator.x_size, generator.y_size)

    total_length = features.index_size
    print("Total length found: ", total_length)
    shared_feature_matrix = np.zeros((shape[0], shape[1], total_length))
    # shared_feature_matrix = np.ctypeslib.as_ctypes(np.zeros((shape[0], shape[1], total_length)))
    # shared_array = sharedctypes.RawArray(shared_feature_matrix._type_, shared_feature_matrix)

    print("\n--- Calculating Feature vector: {} ---\n".format(shared_feature_matrix.shape))

    cached = {}
    # if load_cached or False:
    #     cached = load_feature_cache(features, image_name, window_size)

    for name, feature in iteritems(features.items):
        if hasattr(feature, 'initialize'):
            feature.initialize(generator.image)

    execute_fn = partial(execute_per_set_of_windows, cached=cached, features=features, image_name=image_name, load_cached=load_cached, window_size=window_size)

    # Chunk windows to amount of cpus (creates one extra job for leftovers..)
    windows = [w for w in generator]

    cpu_cnt = cpu_count()

    chunk_size = math.floor(len(windows) / cpu_cnt)

    # Try chunk size for every row of the image
    # Resulting in more cache hits...
    chunk_size = shared_feature_matrix.shape[1]
    if hasattr(feature, 'chunk_size'):
        chunk_size = feature.chunk_size(cpu_cnt, shared_feature_matrix.shape[1])

    windows_chunked = [windows[i:i+chunk_size] for i in range(0, len(windows), chunk_size)]

    total_chunks = len(windows_chunked)

    print("\nTotal chunks to compute: {}, chunk_size: {}".format(total_chunks, chunk_size))
    p = Pool(cpu_cnt, maxtasksperchild=2)
    res = p.map(execute_fn, windows_chunked, chunksize=1)
    p.close()
    p.join()

    to_cache = load_to_cache(features)
    for chunk in res:
        for name, feature_cache_key, x, y, f_indices, feature_array in chunk:
            shared_feature_matrix[x, y, f_indices] = feature_array
            to_cache[name][feature_cache_key] = feature_array
    # shared_feature_matrix = np.ctypeslib.as_array(shared_array)

    # for window in generator:
    #     fill_per_window(window, cached, features, image_name, load_cached, to_cache, window_size)

        # Re-draws a progress bars every 500 windows
        # windows_processed += 1
        # percentage = 1 / total_windows * windows_processed
        # if percentage > 0 and windows_processed % 500 == 0:
        #     draw_progress_bar(percentage)

    cache_calculated_features(features, image_name, to_cache, window_size)

    end = time.time()
    print("Elapsed time extract multiprocessing: {} minutes, start: {}, end: {}".format((end - start) / 60, start, end))

    return shared_feature_matrix




def extract_features_futur2(features: FeatureSet, generator: CellGenerator, load_cached=True, image_name=""):
    start = time.time()
    shape = generator.shape()

    shared_feature_matrix = np.zeros((shape[0], shape[1], 1))
    print("\n--- Calculating Feature vector: {} ---\n".format(shared_feature_matrix.shape))

    for name, feature in iteritems(features.items):
        key = "feature-{feature}-window{window}-image-{image_name}".format(
            image_name=image_name,
            window=(generator.x_size, generator.y_size),
            feature=str(feature),
        )

        feature_matrix = None
        if "Te-" not in str(feature):
            feature_matrix = load_cache(key)

        if feature_matrix is None:
            feature_matrix = compute_feature(feature, generator)

            cache(feature_matrix, key)

        print(shared_feature_matrix.shape, feature_matrix.shape)
        shared_feature_matrix = np.append(shared_feature_matrix, feature_matrix, axis=2)

        # Dirty fix. Would be better to re-use the windows every time so that
        # the windows do not have to be recalculated
        generator = CellGenerator(generator.image, (generator.x_size, generator.y_size))

    end = time.time()
    print("Elapsed time extract multiprocessing: {} minutes, start: {}, end: {}".format((end - start) / 60, start, end))

    return shared_feature_matrix


def timing(f):
    def wrap(*args):
        time1 = time.time()
        ret = f(*args)
        time2 = time.time()
        print('%s function took %0.3f minutes' % (f.__name__, (time2-time1) / 60))
        return ret
    return wrap


@timing
def compute_feature(feature, generator):
    print("\n--- Calculating feature: {} ---\n".format(feature))

    start = time.time()
    if hasattr(feature, 'initialize'):
        data = feature.initialize(generator)
    else:
        raise ValueError("Initialize not implemented")
    end = time.time()
    print("Preparing data cells took {} seconds".format((end - start)))
    shape = generator.shape()

    chunk_size = shape[0]
    cpu_cnt = cpu_count()
    if hasattr(feature, 'chunk_size'):
        chunk_size = feature.chunk_size(cpu_cnt, shape)

    if "Te-" not in str(feature) or True:
        windows_chunked = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
        total_chunks = len(windows_chunked)

        print("\nTotal chunks to compute: {}, chunk_size: {}".format(total_chunks, chunk_size))

        p = Pool(cpu_cnt, maxtasksperchild=1)
        compute_chunk_f = partial(compute_chunk, feature=feature)
        processing_results = p.map(compute_chunk_f, windows_chunked, chunksize=1)
        p.close()
        p.join()
    else:
        print("not using parallel processing...")

        processing_results = [compute_chunk(data, feature), ]



    feature_matrix = np.zeros((shape[0], shape[1], feature.feature_size))
    for coords, chunk_matrix in processing_results:
        load_results_into_matrix(feature_matrix, coords, chunk_matrix)

    return feature_matrix

@njit
def load_results_into_matrix(feature_matrix, coords, chunk_matrix):
    for i in range(coords.shape[0]):
        x = int(coords[i, 0])
        y = int(coords[i, 1])
        fv = chunk_matrix[i, :]
        feature_matrix[x, y, :] = fv


def compute_chunk(chunk, feature):
    start = time.time()

    result = feature(chunk)

    end = time.time()
    delta = (end - start)
    per_block = delta / len(chunk)
    print("Calculating {} Windows took {} seconds each window took: {} (on avg)".format(len(chunk), delta, per_block))
    # print("Memory use: {}".format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))


    return result




def execute_per_set_of_windows(windows, cached, features, image_name, load_cached, window_size):
    start = time.time()
    feature_vectors = []
    for window in windows:
        for name, feature in iteritems(features.items):
            feature_cache_key = make_feature_cache_key(feature, image_name, window, window_size)


            # If we have this feature cached we just load it from there
            # Has to be cached at exact same windows (= same dims)
            # if load_cached and name in cached and feature_cache_key in cached[name]:
            #     feature_array = cached[name][feature_cache_key]
            #     tmp[window.x, window.y, feature.indices] = feature_array
                # feature_vectors.append((window.x, window.y, feature.indices, feature_array))
                #
                # continue

            # Calculate feature and set in feature_vector
            feature_array = feature(window)
            # tmp[window.x, window.y, feature.indices] = feature_array
            # feature_vectors.append((name, feature_cache_key, window.x, window.y, feature.indices, feature_array))

            # Save calculations so we can cache these later
            # to_cache[name][feature_cache_key] = feature_array

    end = time.time()
    delta = (end - start)
    per_block = delta / len(windows)
    print("Calculating {} Windows took {} seconds each window took: {} (on avg) start: {}, end: {}".format(len(windows), delta, per_block, start, end))
    # return feature_vectors

if __name__ == "__main__":
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
    ce = sat_image.canny_edged
    del ce
    generator = CellGenerator(image=sat_image, size=(25, 25))


    feature_set = FeatureSet()
    feature_set.add(Lacunarity(((100, 100),), box_sizes=(5,)))
    extract_features_conc(feature_set, generator, False, image_name)
