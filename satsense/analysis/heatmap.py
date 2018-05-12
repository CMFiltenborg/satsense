import itertools
import math

from satsense import SatelliteImage
from satsense.analysis.plot import load_from_file
from satsense.classification.classifiers import knn_classifier, RF_classifier
from satsense.features import FeatureSet, Pantex
from satsense.features.lacunarity import Lacunarity
from satsense.bands import WORLDVIEW3
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_validate
from sklearn.metrics import confusion_matrix
from satsense.classification.cache import get_project_root
import json
from time import gmtime, strftime
from satsense.classification.model import get_x_matrix, get_y_vector, balance_dataset, create_sift_feature, \
    create_models
from satsense.image import normalize_image, get_rgb_bands
from satsense.performance import jaccard_index_binary_masks
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.model_selection import validation_curve


n_clusters = 32
pantex_window_sizes = ((25, 25), (50, 50), (100, 100), (150, 150))
class_names = {
    0: 'Non-Slum',
    1: 'Slum',
}
images = [
    'section_1',
    'section_2',
    'section_3',
]
show_plots = True
results_path = '{root}/results'.format(root=get_project_root())
current_time = strftime("%Y-%m-%d_%H:%M:%S", gmtime())
base_path = "/home/max/Documents/ai/scriptie/data/Clip"
extension = 'tif'
main_window_size = (1, 1)
percentage_threshold = 0.5
test_size = 0.2
class_ratio = 1.3
feature_set = FeatureSet()
pantex = Pantex(pantex_window_sizes)

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


    # sift = create_sift_feature(sat_image, ((25, 25), (50, 50), (100, 100)), image_name, n_clusters=n_clusters,
    #                            cached=True)

    # lacunarity = create_lacunarity(sat_image, image_name, windows=((25, 25),), cached=True)
    lacunarity = Lacunarity(windows=((100, 100), (200, 200), (300, 300)))
    feature_set.add(lacunarity, "LACUNARITY")
    # feature_set.add(pantex, "PANTEX")
    # feature_set.add(sift, "SIFT")

    classifier = RF_classifier()

    # del sat_image  # Free-up memory

    X = get_x_matrix(sat_image, image_name=image_name, feature_set=feature_set, window_size=main_window_size,
                     cached=True)

    X = np.mean(X, axis=3)

    ds, img, bands = load_from_file(image_file, WORLDVIEW3)
    img = normalize_image(img, bands)
    rgb_img = get_rgb_bands(img, bands)

    plt.imshow(rgb_img)

    binary_mask = X
    show_mask = np.ma.masked_where(binary_mask == 0, binary_mask)
    plt.imshow(show_mask[:, :, 0], cmap='jet', interpolation='none', alpha=0.7)
    plt.title('Binary mask')
    plt.show()
    print('Min {} Max {}'.format(binary_mask.min(), binary_mask.max()))
    print('Len > 0: {}'.format(len(binary_mask[binary_mask > 0])))
    print('Len == 0: {}'.format(len(binary_mask[binary_mask == 0])))

    jaccard_index = jaccard_index_binary_masks(truth_mask[:, :, 0], binary_mask[:, :, 0])
    print("Jaccard index: {}".format(jaccard_index))



    # y, real_mask = get_y_vector(mask_full_path, main_window_size, percentage_threshold, cached=False)
    # X, y = balance_dataset(X, y, class_ratio=class_ratio)
    #
    #
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=None)
    #
    # classifier.fit(X_train, y_train)
    # y_pred = classifier.predict(X_test)
    # accuracy = accuracy_score(y_test, y_pred)
    #
    # X, y, real_mask, groups = create_models(images, feature_set, base_path, main_window_size=main_window_size, percentage_threshold=percentage_threshold, class_ratio=class_ratio, bands=bands)
    # cv = LeaveOneGroupOut()
    # cv_results = cross_validate(classifier, X=X, y=y, groups=groups, return_train_score=True, cv=cv, n_jobs=-1)
    #
    # jaccard_score = jaccard_index_binary_masks(y_test, y_pred)
    # print("Accuracy: {}".format(accuracy))
    # print("Cross validation: ", cv_results)
    # print("Jaccard score: ", jaccard_score)
    #
    # cnf_matrix = confusion_matrix(y_test, y_pred)
    # plots(classifier, X, y, groups, cv, cnf_matrix, show=show_plots, current_time=current_time, results_path=results_path)
    #
    # x_length = math.ceil(sat_image.shape[0] / main_window_size[0])
    # y_length = math.ceil(sat_image.shape[1] / main_window_size[0])
    # feature_mask_shape = (x_length, y_length)
    #
    # y_real_mask = y.reshape(feature_mask_shape)
    # y_pred_mask = y_pred.reshape(feature_mask_shape)
    #
    # result_information = {
    #     'base_satellite_image': image_name,
    #     'classifier': str(classifier),
    #     'results': {
    #         'accuracy': accuracy,
    #         'cv_results': {k: str(v) for k, v in cv_results.items()},
    #         'cnf_matrix': str(cnf_matrix),
    #         'test_size': test_size,
    #         'jaccard_similarity_score': jaccard_score
    #     },
    #     'percentage_threshold': percentage_threshold,
    #     'features': feature_set.string_presentation(),
    #     'main_window_size': main_window_size,
    #     'class_ratio': class_ratio,
    #     'data_characteristics': {
    #         'X.shape': X.shape,
    #         'y.shape': y.shape,
    #         'y[1s]': len(y[y == 1]),
    #         'y[0s]': len(y[y == 0]),
    #         'X_train.shape': X_train.shape,
    #         'X_test.shape': X_test.shape,
    #         'y_train.shape': y_train.shape,
    #         'y_test.shape': y_test.shape,
    #     },
    #     'save_path': "{}/classification_{}.json".format(results_path, current_time),
    #     'classified_at': current_time,
    # }
    # save_classification_results(result_information)
    #
