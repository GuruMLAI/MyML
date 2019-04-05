from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

def cross_validation_score( data, features, label, fold):

    kf = KFold(n_splits=fold, shuffle=True)
    kf.get_n_splits(data)

    for train_index, test_index in kf.split(data):

        X_train, X_test = data.iloc[train_index][features], data.iloc[test_index][features]
        y_train, y_test = data.iloc[train_index][label], data.iloc[test_index][label]


    print('Done.')


