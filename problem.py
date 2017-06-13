import os
import pandas as pd
import rampwf as rw
from sklearn.model_selection import StratifiedShuffleSplit

problem_title = 'Iris classification'
prediction_type = rw.prediction_types.multiclass
workflow = rw.workflows.Classifier()
prediction_labels = ['setosa', 'versicolor', 'virginica']
_target_column_name = 'species'

score_types = [
    rw.score_types.Accuracy(name='acc', n_columns=len(prediction_labels)),
    rw.score_types.ClassificationError(
        name='err', n_columns=len(prediction_labels)),
    rw.score_types.NegativeLogLikelihood(
        name='nll', n_columns=len(prediction_labels)),
    rw.score_types.F1Above(
        name='f1_70', n_columns=len(prediction_labels), threshold=0.7),
]


def get_cv(X, y):
    cv = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=57)
    return cv.split(X, y)


def _read_data(path, f_name):
    data = pd.read_csv(os.path.join(path, 'data', f_name))
    y_array = data[_target_column_name].values
    X_array = data.drop([_target_column_name], axis=1).values
    return X_array, y_array


def get_train_data(path='.'):
    f_name = 'train.csv'
    return _read_data(path, f_name)


def get_test_data(path='.'):
    f_name = 'test.csv'
    return _read_data(path, f_name)
