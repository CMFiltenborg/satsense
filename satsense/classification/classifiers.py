from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn import ensemble
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model.logistic import LogisticRegression

def all_classifiers():
    soft_voting_classifiers = [
        ('RF', RandomForestClassifier(n_jobs=-1)),
        ('GB', ensemble.GradientBoostingClassifier()),
        ('LR', LogisticRegression(n_jobs=-1, solver='saga'))
        # ('GNB', GaussianNB())
    ]
    hard_voting_classifiers = [
        ('RF', RandomForestClassifier(n_jobs=-1)),
        ('GB', ensemble.GradientBoostingClassifier()),
        ('GNB', GaussianNB()),
        ('LR', LogisticRegression(n_jobs=-1, solver='saga'))
    ]
    return [
        # ('BalancedRandomForest', RandomForestClassifier(max_depth=None, class_weight="balanced", n_jobs=-1)),
        # ('RandomForest', RandomForestClassifier(max_depth=None, n_jobs=-1)),
        # ('GradientBoosting', gradient_booster()),
        # ('AdaBoost', AdaBoostClassifier()),
        # ('BalancedSVM', svm.SVC(class_weight='balanced')),
        # ('SVM', svm.SVC()),
        # ('GaussianNB', GaussianNB()),
        # ('LogisticRegression', LogisticRegression(n_jobs=-1, solver='saga')),
        # ('SoftVoting', ensemble.VotingClassifier(
        #     estimators=soft_voting_classifiers,
        #     voting='soft',
        #     n_jobs=-1)
        # ),
        ('HardVoting', ensemble.VotingClassifier(
            estimators=hard_voting_classifiers,
            voting='hard',
            n_jobs=-1)
        ),
    ]

    # yield from (SVM_classifier(), RF_classifier())


def SVM_classifier():
    return svm.SVC()


def RF_classifier():
    return RandomForestClassifier(max_depth=None, class_weight="balanced")


def knn_classifier():
    return KNeighborsClassifier(n_neighbors=3)


def gradient_booster():
    return ensemble.GradientBoostingClassifier()
