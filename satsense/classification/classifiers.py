from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm

def all_classifiers():
    return [RF_classifier(),]
    # yield from (SVM_classifier(), RF_classifier())


def SVM_classifier():
    return svm.SVC()


def RF_classifier():
    return RandomForestClassifier(max_depth=None, random_state=0)


def knn_classifier():
    return KNeighborsClassifier(n_neighbors=3)

