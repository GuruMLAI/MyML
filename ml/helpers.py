from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,RandomForestRegressor,GradientBoostingRegressor
from sklearn.metrics import auc, roc_auc_score, accuracy_score

from ml.utils import cross_validation_score

class FeatureSelection:

    def __init__(self,model=LogisticRegression, mertic='auc', sparsify=True):
        self.model = model
        self.metric = mertic
        self.sparsify = sparsify


    def first_criteria(self, data, features, label, min_threshold=None):

        if self.sparsify:
            data = data.to_sparse(fill_value=0)

        min_threshold_featres = []

        print('Applying the minimum threshold criteria...\n')
        for i in features:
            cv_score = cross_validation_score(self.model, data, [i], label, self.metric)

            if cv_score > min_threshold:
                min_threshold_featres.append(i)

        print('The following {} variables passed the minimum criteria.'.format(len(min_threshold_featres)))
        print('{}\n'.format(min_threshold_featres))

        return min_threshold_featres


    def select_features(self, data, features, label, min_threshold=None):

        if self.sparsify:
            data = data.to_sparse(fill_value=0)

        if min_threshold is not None:
            feature_subset = self.first_criteria(data, features, label, min_threshold)
        else:
            feature_subset = features

        final_metric = 0
        break_val = 0
        rem_features = feature_subset
        sel_features = []

        print('Starting feature selection')

        while break_val == 0 and len(rem_features) > 0:

            loop_variable = ''

            for i in rem_features:

                cv_score = cross_validation_score(self.model, data, sel_features + [i], label, self.metric, fold=5)
                if cv_score > final_metric:
                    final_metric = cv_score
                    loop_variable = i

            if loop_variable == '':
                break_val = 1
            else:
                sel_features = sel_features + [loop_variable]
                rem_features.remove(loop_variable)
                print('Updating the feature list to {} with {} of {}'.format(sel_features, self.metric, final_metric))

        return sel_features