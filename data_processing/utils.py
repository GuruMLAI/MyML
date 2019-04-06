from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import KFold


def metric_calc(model, X, y, metric='auc'):
    if metric == 'auc':
        _metric = roc_auc_score(y, model.predict_proba(X)[:, 1])
    else:
        _metric = accuracy_score(y, [round(value) for value in model.predict_proba(X)[:, 1]])

    return _metric

def cross_validation_score(model, data, features, label, metric='auc', fold=5):

    kf = KFold(n_splits=fold, shuffle=True)
    kf.get_n_splits(data)

    mean_metric = 0
    for train_index, test_index in kf.split(data):

        X_train, X_test = data.iloc[train_index][features], data.iloc[test_index][features]
        y_train, y_test = data.iloc[train_index][label], data.iloc[test_index][label]

        ml = model()
        ml.fit(X_train, y_train)
        #print(y_test)
        #print(ml.predict_proba(X_test))
        #ml_metric = roc_auc_score(y_test, ml.predict_proba(X_test))
        ml_metric = metric_calc(ml, X_test, y_test, metric)

        mean_metric += ml_metric/fold

    print('The mean cross-validation {} score is :{}'.format(metric, mean_metric))

    full_ml = model()
    full_ml.fit(data[features], data[label])
    full_metric = metric_calc(full_ml, data[features], data[label], metric)

    return full_ml, mean_metric


