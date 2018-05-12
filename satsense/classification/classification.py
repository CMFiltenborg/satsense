import itertools
import math

from satsense import SatelliteImage
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
from satsense.performance import jaccard_index_binary_masks
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.model_selection import validation_curve

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


def plots(classifier, X, y, groups, cv, cnf_matrix, results_path, current_time, show=True):
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

    plt.figure()

    plot_learning_curve(estimator=classifier, title='Learning curve', X=X, y=y, groups=groups, ylim=(0.7, 1.01), cv=cv, n_jobs=4)
    # plot_learning_curve(estimator=classifier, X, y, normalize=True,
    #                       title='Normalized confusion matrix')
    plt.savefig("{}/learning_curve_{}.png".format(results_path, current_time))


    if show:
        plt.show()


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, groups=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, groups=groups, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

def plot_validation_curve():
    param_range = np.logspace(-6, -1, 5)
    train_scores, test_scores = validation_curve(
        classifier, X, y, param_name="gamma", param_range=param_range,
        cv=10, scoring="accuracy", n_jobs=1)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.title("Validation Curve with SVM")
    plt.xlabel("$\gamma$")
    plt.ylabel("Score")
    plt.ylim(0.0, 1.1)
    lw = 2
    plt.semilogx(param_range, train_scores_mean, label="Training score",
                 color="darkorange", lw=lw)
    plt.fill_between(param_range, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.2,
                     color="darkorange", lw=lw)
    plt.semilogx(param_range, test_scores_mean, label="Cross-validation score",
                 color="navy", lw=lw)
    plt.fill_between(param_range, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.2,
                     color="navy", lw=lw)
    plt.legend(loc="best")
    plt.show()


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
    'section_2',
    'section_3',
]
show_plots = True
results_path = '{root}/results'.format(root=get_project_root())
current_time = strftime("%Y-%m-%d_%H:%M:%S", gmtime())
base_path = "/home/max/Documents/ai/scriptie/data/Clip"
extension = 'tif'
main_window_size = (25, 25)
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


    sift = create_sift_feature(sat_image, ((25, 25), (50, 50), (100, 100)), image_name, n_clusters=n_clusters,
                               cached=True)

    # lacunarity = create_lacunarity(sat_image, image_name, windows=((25, 25),), cached=True)
    lacunarity = Lacunarity(windows=((10, 10), (20, 20), (30, 30)))
    feature_set.add(lacunarity, "LACUNARITY")
    # feature_set.add(pantex, "PANTEX")
    # feature_set.add(sift, "SIFT")

    classifier = RF_classifier()

    # del sat_image  # Free-up memory

    X = get_x_matrix(sat_image, image_name=image_name, feature_set=feature_set, window_size=main_window_size,
                     cached=True)
    y, real_mask = get_y_vector(mask_full_path, main_window_size, percentage_threshold, cached=False)
    X, y = balance_dataset(X, y, class_ratio=class_ratio)


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=None)

    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    X, y, real_mask, groups = create_models(images, feature_set, base_path, main_window_size=main_window_size, percentage_threshold=percentage_threshold, class_ratio=class_ratio, bands=bands)
    cv = LeaveOneGroupOut()
    cv_results = cross_validate(classifier, X=X, y=y, groups=groups, return_train_score=True, cv=cv, n_jobs=-1)

    jaccard_score = jaccard_index_binary_masks(y_test, y_pred)
    print("Accuracy: {}".format(accuracy))
    print("Cross validation: ", cv_results)
    print("Jaccard score: ", jaccard_score)

    cnf_matrix = confusion_matrix(y_test, y_pred)
    plots(classifier, X, y, groups, cv, cnf_matrix, show=show_plots, current_time=current_time, results_path=results_path)

    x_length = math.ceil(sat_image.shape[0] / main_window_size[0])
    y_length = math.ceil(sat_image.shape[1] / main_window_size[0])
    feature_mask_shape = (x_length, y_length)

    y_real_mask = y.reshape(feature_mask_shape)
    y_pred_mask = y_pred.reshape(feature_mask_shape)

    result_information = {
        'base_satellite_image': image_name,
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
