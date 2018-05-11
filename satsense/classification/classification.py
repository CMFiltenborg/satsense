import itertools
import time
import math
from sklearn.neighbors import KNeighborsClassifier
from satsense import SatelliteImage, extract_features
from satsense.classification.classifiers import knn_classifier
from satsense.features import FeatureSet, Pantex
from satsense.features.lacunarity import create_lacunarity, Lacunarity
from satsense.features.sift import Sift, sift_cluster
from satsense.generators import CellGenerator
from satsense.bands import WORLDVIEW2, MASK_BANDS, WORLDVIEW3
import numpy as np
import gdal
import os
import sys
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from satsense.classification.cache import cache, load_cache, cache_model, cached_model_exists, load_cached_model
import json
from time import gmtime, strftime
from satsense.classification.model import get_x_matrix, get_y_vector, balance_dataset
from satsense.performance import jaccard_index_binary_masks


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def plots(cnf_matrix, results_path, current_time, show=True):
    np.set_printoptions(precision=2)
    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names,
                          title='Confusion matrix, without normalization')
    plt.savefig("{}/confusion_matrix_{}.png".format(results_path, current_time))

    if show:
        plt.show()

    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                          title='Normalized confusion matrix')
    plt.savefig("{}/confusion_matrix_normalized_{}.png".format(results_path, current_time))

    if show:
        plt.show()


def create_sift_feature(sat_image: SatelliteImage, window_sizes, image_name, n_clusters=32, cached=True) -> Sift:
    cache_key = "kmeans-{}".format(image_name)

    if cached and cached_model_exists(cache_key):
        kmeans = load_cached_model(cache_key)
        print("Loaded cached kmeans {}".format(cache_key))
    else:
        print("Computing k-means model")
        kmeans = sift_cluster(sat_image, n_clusters=n_clusters)
        cache_model(kmeans, cache_key)

    feature = Sift(kmeans, windows=(window_sizes))

    return feature


def save_classification_results(results):
    with open(results['save_path'], 'w') as fp:
        json.dump(results, fp, indent=4)


n_clusters = 32
pantex_window_sizes = ((25, 25), (50, 50), (100, 100), (150, 150))
class_names = {
    0: 'Non-Slum',
    1: 'Slum',
}
images = [
    'section_1',
    # 'section_2',
    # 'section_3',
]
show_plots = True
results_path = "/home/max/Documents/ai/scriptie/satsense/results"
current_time = strftime("%Y-%m-%d_%H:%M:%S", gmtime())
base_path = "/home/max/Documents/ai/scriptie/data/Clip"
extension = 'tif'
main_window_size = (25, 25)
percentage_threshold = 0.5
test_size = 0.2
class_ratio = 1.3
feature_set = FeatureSet()
pantex = Pantex(pantex_window_sizes)
full_test = False

for image_name in images:
    image_file = "{base_path}/{image_name}.{extension}".format(
        base_path=base_path,
        image_name=image_name,
        extension=extension
    )
    bands = WORLDVIEW3
    mask_full_path = "{base_path}/{image_name}_masked.tif".format(base_path=base_path, image_name=image_name)
    sat_image = SatelliteImage.load_from_file(image_file, bands)
    sat_image_shape = sat_image.shape


    sift = create_sift_feature(sat_image, ((25, 25), (50, 50), (100, 100)), image_name, n_clusters=n_clusters,
                               cached=True)

    # lacunarity = create_lacunarity(sat_image, image_name, windows=((25, 25),), cached=True)
    lacunarity = Lacunarity(windows=((10, 10), (20, 20), (30, 30)))
    feature_set.add(lacunarity, "LACUNARITY")
    # feature_set.add(pantex, "PANTEX")
    # feature_set.add(sift, "SIFT")

    classifier = knn_classifier()


    # del sat_image  # Free-up memory

    X = get_x_matrix(sat_image, image_name=image_name, feature_set=feature_set, window_size=main_window_size,
                     cached=True)
    y, real_mask = get_y_vector(mask_full_path, main_window_size, percentage_threshold, cached=False)
    X, y = balance_dataset(X, y, class_ratio=class_ratio)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=None)

    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cv_results = cross_validate(classifier, X, y, return_train_score=True, cv=10)
    jaccard_score = jaccard_index_binary_masks(y_test, y_pred)
    print("Accuracy: {}".format(accuracy))
    print("Cross validation: ", cv_results)
    print("Jaccard score: ", jaccard_score)

    cnf_matrix = confusion_matrix(y_test, y_pred)
    plots(cnf_matrix, show=show_plots, current_time=current_time, results_path=results_path)

    # x_length = math.ceil(sat_image.shape[0] / main_window_size[0])
    # y_length = math.ceil(sat_image.shape[1] / main_window_size[0])
    # feature_mask_shape = (x_length, y_length)

    # y_real_mask = y.reshape(feature_mask_shape)
    # y_pred_mask = y_pred.reshape(feature_mask_shape)

    result_information = {
        'satellite_image': image_name,
        'classifier': str(classifier),
        'results': {
            'accuracy': accuracy,
            'cv_results': {k: str(v) for k, v in cv_results.items()},
            'cnf_matrix': str(cnf_matrix),
            'test_size': test_size,
            'jaccard_similarity_score': jaccard_score
        },
        'percentage_threshold': percentage_threshold,
        'features': feature_set.string_presentation(),
        'main_window_size': main_window_size,
        'class_ratio': class_ratio,
        'data_characteristics': {
            'X.shape': X.shape,
            'y.shape': y.shape,
            'y[1s]': len(y[y == 1]),
            'y[0s]': len(y[y == 0]),
            'X_train.shape': X_train.shape,
            'X_test.shape': X_test.shape,
            'y_train.shape': y_train.shape,
            'y_test.shape': y_test.shape,
        },
        'save_path': "{}/classification_{}.json".format(results_path, current_time),
        'classified_at': current_time,
    }
    save_classification_results(result_information)

# Calculate Mean PANTEX/GLCM variance for different classes
