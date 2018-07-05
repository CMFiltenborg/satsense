import itertools
import os

import gdal

from satsense import SatelliteImage
from satsense.features import FeatureSet, Pantex
from satsense.features.lacunarity import Lacunarity
from satsense.bands import WORLDVIEW3
import numpy as np
import matplotlib.pyplot as plt

from satsense.features.sift import sift_cluster, Sift
from satsense.features.texton import texton_cluster, Texton
from satsense.generators import CellGenerator
from satsense.image import normalize_image, get_rgb_bands

plt.switch_backend('agg')
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, matthews_corrcoef
from sklearn.metrics import confusion_matrix
from satsense.util.path import get_project_root, data_path
from time import gmtime, strftime
from satsense.classification.model import create_models, cv_train_test_split_images, get_x_matrix, \
    get_y_vector, \
    save_classification_results, generate_classifiers, generate_feature_scales, generate_oversamplers
from satsense.performance import jaccard_index_binary_masks
import pandas as pd
from sklearn.preprocessing import RobustScaler


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
        save_path = "{}/confusion_matrix_{}.png".format(results_path, current_time)
    else:
        save_path = "{}/confusion_matrix_normalized_{}.png".format(results_path, current_time)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)




def plot_overlap(y_test, y_pred, image_name, image_full_path, test_sat_image, mask_full_path, main_window_size, current_time, results_path):
    # dataset = gdal.Open(mask_full_path, gdal.GA_ReadOnly)
    # array = dataset.ReadAsArray()
    # array = np.min(array, 0)
    # array = array[:, :, np.newaxis]

    # truth_mask = np.where(array > 0, 1, 0)
    # binary_sat_image = SatelliteImage.load_from_file(binary_file_path, bands=mask_bands)
    # binary_sat_image = SatelliteImage(dataset, array, MASK_BANDS)
    generator = CellGenerator(image=test_sat_image, size=main_window_size)

    # result_mask = np.zeros(array.shape, dtype=np.uint8)
    # result_mask = np.zeros(generator.shape())
    # truth_mask = np.zeros(generator.shape())

    # y_pred_im = y_pred[groups == im_num]
    # print("unique y_pred", np.unique(y_pred, return_counts=True))
    # print(y_pred.shape)
    # print(y_pred)
    # print("Gen shape", generator.x_length, generator.y_length, generator.x_length * generator.y_length)
    # print("result mask shape", result_mask.shape)

    print("{} == {}".format(generator.x_length * generator.y_length, y_pred.shape))
    full_scale_pred_mask = np.zeros(
        (test_image_loaded.shape[0], test_image_loaded.shape[1])
    )
    full_scale_truth_mask = np.zeros(
        (test_image_loaded.shape[0], test_image_loaded.shape[1])
    )

    gen_length = len(tuple(i for i in generator))
    # print("{} == {}".format(gen_length, generator.x_length * generator.y_length))
    # print("{} == {}".format(gen_length, y_pred.shape))
    # print("{} == {}".format(gen_length, y_test.shape))
    # assert(gen_length == generator.x_length * generator.y_length)
    generator = CellGenerator(image=test_sat_image, size=main_window_size)

    i = 0
    y_expected = 0
    for window in generator:
        # if i == generator.x_length * generator.y_length:
        #     print("skipping", i, window.x, window.y)
        #     continue
        # y = 0
        # if i < y_pred.shape[0] >= i:
        #     if y_pred[i] == 0:
        #         y = 0
        #     if y_pred[i] == 1:
        #         y = 255
        #
        # y_matrix[window.x, window.y] = y
        # result_mask[window.x_range, window.y_range, 0] = y
        # i += 1
        # if y > 0:
        #     y_expected += 30 * 30

        # truth_mask[window.x, window.y] = y_test[i]
        # result_mask[window.x, window.y] = y_pred[i]
        full_scale_pred_mask[window.x_range, window.y_range] = y_pred[i]
        full_scale_truth_mask[window.x_range, window.y_range] = y_test[i]

        i += 1


    result_mask = np.reshape(y_pred, generator.shape())
    truth_mask = np.reshape(y_test, generator.shape())


    # print("{} == {}".format(y_expected, len(result_mask[result_mask > 0])))
    # print("Total iterations", i)
    # print("Y_matrix counts", np.unique(y_matrix, return_counts=True))
    # print("Counts:", np.unique(result_mask, return_counts=True))
    # print("result_mask[1s]", len(result_mask[result_mask == 1]))
    # print("result_mask[0s]", len(result_mask[result_mask == 0]))

    ds, img, bands = load_from_file(image_full_path, WORLDVIEW3)
    img = normalize_image(img, bands)
    rgb_img = get_rgb_bands(img, bands)
    # grayscale = get_grayscale_image(img, bands)

    plt.figure()
    plt.axis('off')
    plt.imshow(rgb_img)
    # plt.imshow(np.zeros(rgb_img.shape)[:, :, 0], cmap='gray')
    # plt.imshow(grayscale, cmap='gray')


    show_mask = np.ma.masked_where(full_scale_pred_mask == 0, full_scale_pred_mask)
    plt.imshow(show_mask, cmap='jet', interpolation='none', alpha=1.0)
    # plt.title('Binary mask')

    plt.savefig("{}/classification_jaccard_results_{}_{}.png".format(results_path, image_name, current_time))
    plt.show()

    plt.figure()
    plt.axis('off')
    plt.imshow(result_mask, cmap='jet', interpolation='none', alpha=1.0)

    plt.savefig("{}/classification_jaccard_mask_results_{}_{}.png".format(results_path, image_name, current_time))
    plt.show()

    # print('Min {} Max {}'.format(result_mask.min(), result_mask.max()))
    # print('Len > 0: {}'.format(len(result_mask[result_mask > 0])))
    # print('Len == 0: {}'.format(len(result_mask[result_mask == 0])))

    jaccard_index_main_window_scale = jaccard_index_binary_masks(truth_mask, result_mask)
    jaccard_index_full_scale = jaccard_index_binary_masks(full_scale_truth_mask, full_scale_pred_mask)


    # print("Jaccard index: {}".format(jaccard_index_main_window_scale))

    return jaccard_index_main_window_scale, jaccard_index_full_scale

def jaccard_similarity(y_truth, y_pred):
    y_truth_slum = y_truth[y_truth == 1]
    y_pred_slum = y_pred[y_truth == 1]

    return np.intersect1d(y_truth_slum, y_pred_slum).size / np.union1d(y_truth_slum, y_pred_slum).size



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
# feature_set = FeatureSet()
# lacunarity = Lacunarity(windows=((100, 100), (200, 200), (300, 300)))
# feature_set.add(lacunarity, "LACUNARITY")
# feature_set.add(pantex, "PANTEX")

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


results = {}
feature_names = [
    ("LACUNARITY", (50, 300, 500)),
    ("PANTEX", (100, 300, 500)),
    ("SIFT", (100, 200, 300)),
    ("TEXTON", (50, 100, 200)),
]

# feature_names = [
#     ("LACUNARITY", (50,)),
    # ("PANTEX", (100,)),
    # ("SIFT", (100,)),
    # ("TEXTON", (50,)),
# ]
scaler_name = "RobustScaler"
scaled = True

cv_test = 0

# for feature_name, feature_scales in feature_names:
for n in (0,):
    feature_name = "ALL"
    results_cols = ['cv_test', 'feature_name', 'classifier_name', 'classifier', 'feature_set', 'test_image', 'feature_scale', 'jaccard_index', 'accuracy', 'precision', 'recall', 'normalized_cnf',]
    results_frame = pd.DataFrame(columns=results_cols)

    df_path = '{root}/results/jaccard'.format(root=get_project_root())
    csv_file = '{}/{}_df_optimization.csv'.format(df_path, feature_name)
    if os.path.exists(csv_file):
        results_frame = pd.read_csv(csv_file)

    # for feature_scale in generate_feature_scales(feature_scales):
    for main_window_size in ((10, 10,),):
        # print("Running feature scale: {}".format(feature_scale))

        for test_image, train_images in cv_train_test_split_images(images):

            print("test im:{} train_ims:{}".format(test_image, train_images))

            image_name = test_image
            mask_full_path = "{base_path}/{image_name}_masked.tif".format(base_path=base_path, image_name=image_name)
            test_image_loaded = load_image(test_image)

            cached = True
            feature_set = FeatureSet()
            for feature_name, feature_scale in feature_names:
                feature_scale = tuple((fc, fc) for fc in feature_scale)

                if feature_name == "SIFT":
                    sift_clusters = sift_cluster(map(load_image, train_images))
                    # for fc in feature_scale:
                    sift = Sift(sift_clusters, windows=feature_scale)
                    feature_set.add(sift)
                    cached = False
                if feature_name == "TEXTON":
                    texton_clusters = texton_cluster(map(load_image, train_images))
                    # for fc in feature_scale:
                    texton = Texton(texton_clusters, windows=feature_scale)
                    feature_set.add(texton)
                    cached = False
                if feature_name == "PANTEX":
                    # for fc in feature_scale:
                    pantex = Pantex(windows=feature_scale)
                    feature_set.add(pantex)
                if feature_name == "LACUNARITY":
                    # for fc in feature_scale:
                    lacunarity = Lacunarity(windows=feature_scale)
                    feature_set.add(lacunarity)

            feature_name = "ALL"
            # if (
            #         ((results_frame['feature_set'] == str(feature_set)) &
            #         (results_frame['test_image'] == test_image)).any() &
            #         (results_frame['main_window_size'] == test_image)).any()
            #     ):
            #     print("Skipping {}".format(str(feature_set)))
            #     continue

            # texton = create_texton_feature(sat_image, ((25, 25), (50, 50), (100, 100)), image_name, n_clusters=n_clusters, cached=True)


            plt.close('all')
            print("Running feature set {}, image {}".format(feature_set, image_name))
            results_path = '{root}/results/jaccard/{fs}'.format(root=get_project_root(), fs=str(feature_set))
            try:
                os.makedirs(os.path.dirname(results_path + '/'), exist_ok=True)
            except OSError:
                pass

            X_test = get_x_matrix(test_image_loaded, image_name=image_name, feature_set=feature_set, window_size=main_window_size, cached=cached)
            y_test, real_mask = get_y_vector(mask_full_path, main_window_size, percentage_threshold, cached=False)

            # X, y = balance_dataset(X, y, class_ratio=class_ratio)
            # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=None)

            X_train, y_train, real_mask, groups_train = create_models(train_images, feature_set, base_path,
                                                                      main_window_size=main_window_size,
                                                                      percentage_threshold=percentage_threshold,
                                                                      class_ratio=class_ratio, bands=bands,
                                                                      cached=cached)

            for oversampler_name, oversampler in generate_oversamplers():
                for classifier_name, classifier in generate_classifiers():
                    print("Running clf: {}, oversampler: {}, scaler: {}".format(classifier_name, oversampler_name, scaler_name))
                    if oversampler is not None:
                        if scaled:
                            robust_scaler = RobustScaler()
                            X_train_scaled = robust_scaler.fit_transform(X_train)
                            X_test_scaled = robust_scaler.transform(X_test)
                            X_train_resampled, y_train_resampled = oversampler.fit_sample(X_train_scaled, y_train)
                        else:
                            X_train_resampled, y_train_resampled = oversampler.fit_sample(X_train, y_train)
                        classifier.fit(X_train_resampled, y_train_resampled)
                    else:
                        classifier.fit(X_train, y_train)

                    if not scaled:
                        y_pred = classifier.predict(X_test)
                    else:
                        y_pred = classifier.predict(X_test_scaled)



                    accuracy = accuracy_score(y_test, y_pred)
                    recall = recall_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred)
                    f1 = f1_score(y_test, y_pred)
                    matthews_score = matthews_corrcoef(y_test, y_pred)
                    print("test_im: {}, train_ims:{}".format(test_image, train_images))
                    print("Oversampler {}, classifier {}".format(oversampler_name, classifier_name))
                    print("Accuracy: {}".format(accuracy))
                    print("matthews score: {}".format(matthews_score))
                    print("precision: {}".format(precision))
                    print("f1: {}".format(f1))
                    current_time = strftime("%Y-%m-%d_%H:%M:%S", gmtime())

                    cnf_matrix = confusion_matrix(y_test, y_pred)
                    cnf_matrix_normalized = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
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
                    jaccard_score, jaccard_score_full_scale = plot_overlap(y_test, y_pred, image_name, image_file, test_image_loaded, mask_full_path, main_window_size, current_time, results_path)

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
                            'precision': precision,
                            'recall': recall,
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

                    if oversampler is not None:
                        nr_train_examples = y_train_resampled.shape[0]
                        nr_train_non_slum_examples = y_train_resampled[y_train_resampled == 0].shape[0]
                        nr_train_slum_examples = y_train_resampled[y_train_resampled == 1].shape[0]
                    else:
                        nr_train_examples = y_train.shape[0]
                        nr_train_non_slum_examples = y_train[y_train == 0].shape[0]
                        nr_train_slum_examples = y_train[y_train == 1].shape[0]

                    results_row = pd.Series({
                        'cv_test': cv_test,
                        'feature_name': feature_name,
                        'classifier_name': classifier_name,
                        'classifier': str(classifier),
                        'feature_set': str(feature_set),
                        'test_image': test_image,
                        'train_ims': train_images,
                        'feature_scale': None,
                        'jaccard_index': jaccard_score,
                        'jaccard_index_full_scale': jaccard_score_full_scale,
                        'accuracy': accuracy,
                        'f1_score': f1,
                        'matthews_score': matthews_score,
                        'precision': precision,
                        'recall': recall,
                        'normalized_cnf': cnf_matrix_normalized,
                        'oversampled': oversampler is not None,
                        'oversampler': oversampler,
                        'oversampler_name': oversampler_name,
                        'train_examples': nr_train_examples,
                        'train_non-slum_examples': nr_train_non_slum_examples,
                        'train_slum_examples': nr_train_slum_examples,
                        'test_examples': y_test.shape[0],
                        'test_non-slum_examples': y_test[y_test == 0].shape[0],
                        'test_slum_examples': y_test[y_test == 1].shape[0],
                        'main_window_size': main_window_size,
                        'current_time': current_time,
                        'scaler': scaler_name,
                        'scaled': scaled,
                    })
                    save_classification_results(result_information, save_path)
                    results_frame = results_frame.append(results_row, ignore_index=True)
                    # print(results_frame)

                    df_path = '{root}/results/jaccard'.format(root=get_project_root())
                    results_frame.to_csv('{}/{}_df_optimization.csv'.format(df_path, feature_name))
                    print("Written frame to '{}'".format(df_path))
        cv_test += 1


# save_path = "{}/best_result_classification_{}.json".format(results_path, current_time)
# save_classification_results(best_result, save_path)
