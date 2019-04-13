import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import KFold


def metric_calc(model, X, y, metric='auc'):
    if metric == 'auc':
        _metric = roc_auc_score(y, model.predict_proba(X)[:, 1])
    else:
        _metric = accuracy_score(y, [round(value) for value in model.predict_proba(X)[:, 1]])

    return _metric


def cross_validation_score(model, data, features, label, metric='auc', fold=None):

    if fold is None:
        X_train = np.array(data.loc[:,features]).reshape(-1, len(features))
        y_train = np.array(data.loc[:,label]).reshape(-1, )

        model.fit(X_train, y_train)
        mean_metric = metric_calc(model, X_train, y_train, metric)

    else:

        kf = KFold(n_splits=fold, shuffle=True)
        kf.get_n_splits(data)

        mean_metric = 0
        for train_index, test_index in kf.split(data):

            X_train = np.array(data.iloc[train_index][features]).reshape(-1,len(features))
            X_test = np.array(data.iloc[test_index][features]).reshape(-1,len(features))
            y_train = np.array(data.iloc[train_index][label]).reshape(-1,)
            y_test = np.array(data.iloc[test_index][label]).reshape(-1,)

            model.fit(X_train, y_train)

            ml_metric = metric_calc(model, X_test, y_test, metric)

            mean_metric += ml_metric/fold

    return mean_metric


