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
plt.switch_backend('agg')
from satsense.util.path import data_path, get_project_root
from scipy.stats import norm
from sklearn.neighbors import KernelDensity
from scipy import stats, integrate

sns.set(color_codes=True)
sns.set_context("paper")

# sns.set(style="whitegrid", rc={'legend.frameon':True})
current_palette = sns.color_palette("Blues")
sns.set_palette(current_palette)
sns.set(style="whitegrid", rc={'legend.frameon':True})

# feature_window_sizes = [(25, 25), (50, 50), (100, 100), (200, 200), (300, 300), (500, 500)]
# feature_window_sizes = [(200, 200)]
feature_window_sizes = [(100, 100), (200, 200), (300, 300), (500, 500)]
feature_name = 'Lacunarity'

main_window_size = (10, 10)
base_path = data_path()

results_path = '{root}/results'.format(root=get_project_root())

extension = 'tif'
balanced = False
images = [
    'section_1',
    'section_2',
    'section_3',
]
class_ratio = 1.0
bands = WORLDVIEW3


def plot_boxplot(X, y, image_name, plot_title, feature_name="GLCM variance"):
    print(X.shape, y.shape)
    df = pd.DataFrame({'X': X.ravel(), 'y': y})
    df['class'] = df['y'].map(lambda x: "Slum" if x == 1 else "Non-slum")
    df[feature_name] = df['X']
    plt.figure()
    # plt.title(plot_title)
    # plt.axis('off')
    sns.boxplot(x='class', y=feature_name, data=df)

    if balanced:
        plt.savefig(results_path + "/{feature_name}/boxplot_balanced_{image_name}_{plot_title}.png".format(
            image_name=image_name,
            plot_title=plot_title,
            feature_name=feature_name.replace(" ", ""),
        ))
    else:
        plt.savefig(results_path + "/{feature_name}/boxplot_{image_name}_{plot_title}.png".format(
            image_name=image_name,
            plot_title=plot_title,
            feature_name=feature_name.replace(" ", ""),
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

    df_slum = pd.DataFrame({'Slum': X_ones.flatten()})
    df_non_slum = pd.DataFrame({'Non-slum': X_zeros.flatten()})
    # df = pd.DataFrame({'Slum': X_ones.flatten(), 'Non-slum': X_zeros.flatten()})

    plt.figure()
    plt.title(plot_title)
    balanced_str = "balanced" if balanced else "not-balanced"
    for df, col in ((df_slum, 'Slum'), (df_non_slum, 'Non-slum')):
        # sns.distplot(df[col], ax=axes[ax_pos])
        ax = sns.kdeplot(df[col], shade=True)

    plt.savefig(results_path + "/{feature_name}/distribution_{balanced}_{image_name}_{plot_title}.png".format(
        image_name=image_name,
        plot_title=plot_title,
        feature_name=feature_name.replace(" ", ""),
        balanced=balanced_str
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
            plot_title = '{} with scale {}, box size {}'.format(feature_name, window_size, box_size)
            # plot_title = 'Lacunarity with window {} with box_size {}'.format(window_size, box_size)
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
    feature_name=feature_name.replace(" ", "")
))
plt.show()
