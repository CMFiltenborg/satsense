from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


def get_classifier():
    return RF_classifier()


def RF_classifier():
    return RandomForestClassifier(max_depth=None, random_state=0)


def knn_classifier():
    return KNeighborsClassifier(n_neighbors=3)

