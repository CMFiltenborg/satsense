from satsense import SatelliteImage, extract_features
from satsense.features import FeatureSet, Pantex
from satsense.features.lacunarity import Lacunarity
from satsense.generators import CellGenerator
from satsense.bands import WORLDVIEW2, MASK_BANDS, WORLDVIEW3
from satsense.classification.model import get_y_vector, get_x_matrix, balance_dataset
import numpy as np
import gdal
import os
import sys
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.neighbors import KernelDensity
from scipy import stats, integrate

sns.set(color_codes=True)

feature_window_sizes = [(25, 25), (50, 50), (100, 100), (200, 200), (300, 300)]
# feature_window_sizes = [(100, 100), (200, 200), (300, 300)]
feature_name = 'lacunarity'

main_window_size = (30, 30)
base_path = "/home/max/Documents/ai/scriptie/data/Clip"
results_path = "/home/max/Documents/ai/scriptie/satsense/results"
extension = 'tif'
balanced = True
images = [
    'section_1',
    'section_2',
    'section_3',
]
class_ratio = 1.0
bands = WORLDVIEW3


def plot_boxplot(X, y, image_name, plot_title, feature_name="pantex"):
    df = pd.DataFrame({'X': X.ravel(), 'y': y})
    df['y'] = df['y'].map(lambda x: "Slum" if x == 1 else "Non-slum")
    sns.boxplot(x='y', y='X', data=df)

    plt.figure()
    plt.title(plot_title)
    if balanced:
        plt.savefig(results_path + "/{feature_name}/boxplot_balanced_{image_name}_{plot_title}.png".format(
            image_name=image_name,
            plot_title=plot_title,
            feature_name=feature_name,
        ))
    else:
        plt.savefig(results_path + "/{feature_name}/boxplot_{image_name}_{plot_title}.png".format(
            image_name=image_name,
            plot_title=plot_title,
            feature_name=feature_name,
        ))
    plt.show()


def plot_distribution_subplots(X, y, image_name, ax_pos):
    X_zeros = X[y == 0, :]
    y_zeros = y[y == 0]
    X_ones = X[y == 1, :]
    y_ones = y[y == 1]
    df = pd.DataFrame({'Slum': X_ones.flatten(), 'Non-slum': X_zeros.flatten()})

    legend = False
    if ax_pos == (0, 0):
        legend = True

    for col in ['Slum', 'Non-slum']:
        # sns.distplot(df[col], ax=axes[ax_pos])
        sns.kdeplot(df[col], shade=True, ax=axes[ax_pos], legend=legend)


def plot_distribution(X, y, image_name, plot_title, feature_name):
    X_zeros = X[y == 0, :]
    y_zeros = y[y == 0]
    X_ones = X[y == 1, :]
    y_ones = y[y == 1]
    df = pd.DataFrame({'Slum': X_ones.flatten(), 'Non-slum': X_zeros.flatten()})

    plt.figure()
    plt.title(plot_title)
    for col in ['Slum', 'Non-slum']:
        # sns.distplot(df[col], ax=axes[ax_pos])
        ax = sns.kdeplot(df[col], shade=True)

    plt.savefig(results_path + "/{feature_name}/distribution_{image_name}_{plot_title}.png".format(
        image_name=image_name,
        plot_title=plot_title,
        feature_name=feature_name
    ))
    # ax.set(xlabel='common xlabel', ylabel='common ylabel')
    plt.show()

# f, axes = plt.subplots(len(images), len(feature_window_sizes), sharex='col', sharey='row', figsize=(15, 15))
for i, image_name in enumerate(images):
    image_file = "{base_path}/{image_name}.{extension}".format(
        base_path=base_path,
        image_name=image_name,
        extension=extension
    )
    mask_full_path = "{base_path}/{image_name}_masked.tif".format(base_path=base_path, image_name=image_name)
    sat_image = SatelliteImage.load_from_file(image_file, bands)
    sat_image_shape = sat_image.shape

    for j, window_size in enumerate(feature_window_sizes):
        for box_size in (10, 20, 30):
            # plot_title = 'GLCM PanTex with window {}'.format(window_size)
            plot_title = 'Lacunarity with window {} with box_size {}'.format(window_size, box_size)
            feature_set = FeatureSet()
            feature_sizes = (window_size,)
            lacunarity = Lacunarity(feature_sizes, (box_size,))
            feature_set.add(lacunarity)

            # pantex = Pantex(feature_sizes)
            # feature_set.add(pantex, "PANTEX")

            X = get_x_matrix(sat_image=sat_image, feature_set=feature_set, window_size=main_window_size, cached=False,
                             image_name=image_name)
            y, _ = get_y_vector(mask_full_path, main_window_size, percentage_threshold=0.5, cached=False)
            if balanced:
                X, y = balance_dataset(X, y, class_ratio=class_ratio)

            print('y[1s]: {}'.format(len(y[y == 1])))
            print('y[0s]: {}'.format(len(y[y == 0])))

            plot_boxplot(X, y, image_name, plot_title=plot_title, feature_name=feature_name)
            # plot_distribution_subplots(X, y, image_name, (i, j))
            plot_distribution(X, y, image_name, plot_title, feature_name=feature_name)

plot_title = 'Lacunarity'
plt.savefig(results_path + "/{feature_name}/distribution_{plot_title}.png".format(
    plot_title=plot_title,
    feature_name=feature_name
))
plt.show()
