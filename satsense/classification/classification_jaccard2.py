import itertools
import os

import gdal

from satsense import SatelliteImage
from satsense.features import FeatureSet, Pantex
from satsense.features.lacunarity import Lacunarity
from satsense.bands import WORLDVIEW3, MASK_BANDS
import numpy as np
import matplotlib.pyplot as plt

from satsense.features.sift import sift_cluster, Sift
from satsense.features.texton import texton_cluster, Texton
from satsense.generators import CellGenerator
from satsense.image import normalize_image, get_rgb_bands, get_grayscale_image

plt.switch_backend('agg')
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from satsense.util.path import get_project_root, data_path
from time import gmtime, strftime
from satsense.classification.model import create_models, generate_tests, cv_train_test_split_images, get_x_matrix, get_y_vector, \
    save_classification_results
from satsense.performance import jaccard_index_binary_masks


def load_from_file(path, bands):
    """
    Loads the specified path from file and loads the bands into a numpy array
    @returns dataset The raw gdal dataset
            image The image loaded as a numpy array
    """
    dataset = gdal.Open(path, gdal.GA_ReadOnly)
    array = dataset.ReadAsArray()

    if len(array.shape) == 3:
        # The bands column is in the first position, but we want it last
        array = np.rollaxis(array, 0, 3)
    elif len(array.shape) == 2:
        # This image seems to have one band, so we add an axis for ease
        # of use in the rest of the library
        array = array[:, :, np.newaxis]

    image = array.astype('float32')

    return dataset, image, bands

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure()

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
    if not normalize:
        plt.savefig("{}/confusion_matrix_{}.png".format(results_path, current_time))
    else:
        plt.savefig("{}/confusion_matrix_normalized_{}.png".format(results_path, current_time))




def plot_overlap(y_pred, image_name, image_full_path, mask_full_path, main_window_size, current_time, results_path):
    dataset = gdal.Open(mask_full_path, gdal.GA_ReadOnly)
    array = dataset.ReadAsArray()
    array = np.min(array, 0)
    array = array[:, :, np.newaxis]

    truth_mask = np.where(array > 0, 1, 0)

    # binary_sat_image = SatelliteImage.load_from_file(binary_file_path, bands=mask_bands)
    binary_sat_image = SatelliteImage(dataset, array, MASK_BANDS)
    generator = CellGenerator(image=binary_sat_image, size=main_window_size)

    result_mask = np.zeros(array.shape, dtype=np.uint8)
    y_matrix = np.zeros(generator.shape())
    # y_pred_im = y_pred[groups == im_num]
    print("unique y_pred", np.unique(y_pred, return_counts=True))
    print(y_pred.shape)
    print(y_pred)
    print("Gen shape", generator.x_length, generator.y_length, generator.x_length * generator.y_length)
    print("result mask shape", result_mask.shape)

    print("{} == {}".format(generator.x_length * generator.y_length, y_pred.shape))

    i = 0
    y_expected = 0
    for window in generator:

        y = 0
        if i < y_pred.shape[0] >= i:
            if y_pred[i] == 0:
                y = 0
            if y_pred[i] == 1:
                y = 255

        y_matrix[window.x, window.y] = y
        result_mask[window.x_range, window.y_range, 0] = y
        i += 1
        if y > 0:
            y_expected += 30 * 30


    print("{} == {}".format(y_expected, len(result_mask[result_mask > 0])))
    print("Total iterations", i)
    print("Y_matrix counts", np.unique(y_matrix, return_counts=True))
    print("Counts:", np.unique(result_mask, return_counts=True))
    print("result_mask[255s]", len(result_mask[result_mask == 255]))
    print("result_mask[0s]", len(result_mask[result_mask == 0]))

    ds, img, bands = load_from_file(image_full_path, WORLDVIEW3)
    img = normalize_image(img, bands)
    rgb_img = get_rgb_bands(img, bands)
    grayscale = get_grayscale_image(img, bands)

    plt.figure()
    plt.axis('off')
    plt.imshow(rgb_img)
    # plt.imshow(np.zeros(rgb_img.shape)[:, :, 0], cmap='gray')
    # plt.imshow(grayscale, cmap='gray')



    binary_mask = result_mask
    show_mask = np.ma.masked_where(binary_mask == 0, binary_mask)
    plt.imshow(show_mask[:, :, 0], cmap='jet', interpolation='none', alpha=1.0)
    # plt.title('Binary mask')

    plt.savefig("{}/classification_jaccard_results_{}_{}.png".format(results_path, image_name, current_time))
    plt.show()

    plt.figure()
    plt.axis('off')
    plt.imshow(binary_mask[:, :, 0], cmap='jet', interpolation='none', alpha=1.0)

    plt.savefig("{}/classification_jaccard_mask_results_{}_{}.png".format(results_path, image_name, current_time))
    plt.show()

    print('Min {} Max {}'.format(binary_mask.min(), binary_mask.max()))
    print('Len > 0: {}'.format(len(binary_mask[binary_mask > 0])))
    print('Len == 0: {}'.format(len(binary_mask[binary_mask == 0])))

    jaccard_index = jaccard_index_binary_masks(truth_mask[:, :, 0], binary_mask[:, :, 0])
    print("Jaccard index: {}".format(jaccard_index))

    return jaccard_index


n_clusters = 32
pantex_window_sizes = ((100, 100), (200, 200), (300, 300))
class_names = {
    0: 'Non-Slum',
    1: 'Slum',
}
images = [
    'section_1',
    'section_2',
    'section_3',
]
show_plots = False
base_path = data_path()
extension = 'tif'
main_window_size = (30, 30)
percentage_threshold = 0.5
class_ratio = 2.0
feature_set = FeatureSet()
pantex = Pantex(pantex_window_sizes)
lacunarity = Lacunarity(windows=((100, 100), (200, 200), (300, 300)))
feature_set.add(lacunarity, "LACUNARITY")
feature_set.add(pantex, "PANTEX")
best_score = 0
bands = WORLDVIEW3

def load_image(image_name):
    bands = WORLDVIEW3
    base_path = data_path()
    image_file = "{base_path}/{image_name}.{extension}".format(
        base_path=base_path,
        image_name=image_name,
        extension=extension
    )

    return SatelliteImage.load_from_file(image_file, bands)


for test_image, train_images in cv_train_test_split_images(images):

    image_name = test_image
    mask_full_path = "{base_path}/{image_name}_masked.tif".format(base_path=base_path, image_name=image_name)
    test_image_loaded = load_image(test_image)

    # sift = create_sift_feature(sat_image, , image_name, n_clusters=n_clusters,
    #                            cached=True)
    sift_clusters = sift_cluster(map(load_image, train_images))
    sift = Sift(sift_clusters, windows=((25, 25), (50, 50), (100, 100)))


    # texton = create_texton_feature(sat_image, ((25, 25), (50, 50), (100, 100)), image_name, n_clusters=n_clusters, cached=True)
    texton_clusters = texton_cluster(map(load_image, train_images))
    texton = Texton(texton_clusters, windows=((25, 25), (50, 50), (100, 100)))

    feature_set.add(sift, "SIFT")
    feature_set.add(texton, "TEXTON")

    for feature_set, classifier in generate_tests((texton, sift, pantex, lacunarity)):
        plt.close('all')
        print("Running feature set {}, image {}, classifier {}".format(feature_set, image_name, str(classifier)))
        results_path = '{root}/results/jaccard/{fs}'.format(root=get_project_root(), fs=str(feature_set))
        os.makedirs(os.path.dirname(results_path + '/'), exist_ok=True)

        X_test = get_x_matrix(test_image_loaded, image_name=image_name, feature_set=feature_set, window_size=main_window_size,
                              cached=True)
        y_test, real_mask = get_y_vector(mask_full_path, main_window_size, percentage_threshold, cached=False)

        # X, y = balance_dataset(X, y, class_ratio=class_ratio)
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=None)


        X_train, y_train, real_mask, groups_train = create_models(train_images, feature_set, base_path, main_window_size=main_window_size, percentage_threshold=percentage_threshold, class_ratio=class_ratio, bands=bands)

        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)

        # print(X.shape)
        # print(y.shape)
        # print("unique y", np.unique(y, return_counts=True))

        # cv = LeaveOneGroupOut()
        # y_pred = cross_val_predict(classifier, X=X, y=y, groups=groups, cv=cv, n_jobs=8)
        # cv_results = cross_validate(classifier, X=X, y=y, groups=groups, return_train_score=True, cv=cv, n_jobs=-1)

        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy: {}".format(accuracy))
        # print("Cross validation: ", cv_results)
        current_time = strftime("%Y-%m-%d_%H:%M:%S", gmtime())

        cnf_matrix = confusion_matrix(y_test, y_pred)
        plot_confusion_matrix(cnf_matrix, classes=class_names,
                              title='Confusion matrix, without normalization')
        plot_confusion_matrix(cnf_matrix, classes=class_names,
                              normalize=True,
                              title='Normalized confusion matrix')

        # plots(classifier, X, y, groups, cv, cnf_matrix, show=show_plots, current_time=current_time, results_path=results_path)
        image_file = "{base_path}/{image_name}.{extension}".format(
            base_path=base_path,
            image_name=image_name,
            extension=extension
        )
        jaccard_score = plot_overlap(y_pred, image_name, image_file, mask_full_path, main_window_size, current_time, results_path)

        # x_length = math.ceil(sat_image.shape[0] / main_window_size[0])
        # y_length = math.ceil(sat_image.shape[1] / main_window_size[0])
        # feature_mask_shape = (x_length, y_length)

        # y_real_mask = y.reshape(feature_mask_shape)
        # y_pred_mask = y_pred.reshape(feature_mask_shape)
        # mean_cv_score = np.mean(cv_results['test_score'])

        save_path = "{}/classification_{}.json".format(results_path, current_time)
        result_information = {
            'base_satellite_image': image_name,
            'classifier': str(classifier),
            'results': {
                # 'mean_cv_score': mean_cv_score,
                'accuracy': accuracy,
                # 'cv_results': {k: str(v) for k, v in cv_results.items()},
                'cnf_matrix': str(cnf_matrix),
                'jaccard_similarity_score': jaccard_score
            },
            'percentage_threshold': percentage_threshold,
            'features': feature_set.string_presentation(),
            'main_window_size': main_window_size,
            'class_ratio': class_ratio,
            'data_characteristics': {
                # 'X.shape': X_test.shape,
                # 'y.shape': y_test.shape,
                'y_test[1s]': len(y_test[y_test == 1]),
                'y_test[0s]': len(y_test[y_test == 0]),
                'X_train.shape': X_train.shape,
                'X_test.shape': X_test.shape,
                'y_train.shape': y_train.shape,
                'y_test.shape': y_test.shape,
            },
            'save_path': save_path,
            'classified_at': current_time,
        }

        # if best_score < mean_cv_score:
        #     best_score = mean_cv_score
        #     best_result = result_information

        save_classification_results(result_information, save_path)

# save_path = "{}/best_result_classification_{}.json".format(results_path, current_time)
# save_classification_results(best_result, save_path)
