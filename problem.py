import os
import pandas as pd
import rampwf as rw
from sklearn.model_selection import StratifiedShuffleSplit

problem_title = 'Iris classification'
prediction_type = rw.prediction_types.multiclass
workflow = rw.workflows.Classifier()
prediction_labels = ['setosa', 'versicolor', 'virginica']

score_types = [
    rw.score_types.Accuracy(name='acc', n_columns=len(prediction_labels)),
    rw.score_types.ClassificationError(
        name='err', n_columns=len(prediction_labels)),
    rw.score_types.NegativeLogLikelihood(
        name='nll', n_columns=len(prediction_labels)),
]


def get_train_data(path='.'):
    data = pd.read_csv(os.path.join(path, 'public_data', 'public_train.csv'))
    target_column_name = 'species'
    y_array = data[target_column_name].values
    X_array = data.drop([target_column_name], axis=1).values
    return X_array, y_array


def get_cv(X, y):
    cv = StratifiedShuffleSplit(n_splits=2, test_size=0.2, random_state=57)
    return cv.split(X, y)
