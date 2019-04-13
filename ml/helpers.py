from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,RandomForestRegressor,GradientBoostingRegressor
from sklearn.metrics import auc, roc_auc_score, accuracy_score

from ml.utils import cross_validation_score

class FeatureSelector:

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


    def add_new_feature(self, data, cur_features, rem_features, label, cur_metric):

        new_feature = ''
        break_val = 0

        for i in rem_features:

            cv_score = cross_validation_score(self.model, data, cur_features + [i], label, self.metric, fold=5)
            if cv_score > cur_metric:
                cur_metric = cv_score
                new_feature = i

        if new_feature == '':
            break_val = 1
            print('Fearure selection complete. No more features to select.')
            print('The final feature list is {} with {} of {}'.format(cur_features, self.metric, cur_metric))
        else:
            cur_features = cur_features + [new_feature]
            rem_features.remove(new_feature)
            print('Updating the feature list to {} with {} of {}'.format(cur_features, self.metric, cur_metric))

        return cur_features, rem_features, cur_metric, break_val, new_feature


    def remove_one_feature(self, data, cur_features, new_feature, label, cur_metric):

        removed_feature = ''
        no_change_value = 0
        remove_candidates = [x for x in cur_features if x is not new_feature]

        for i in remove_candidates:
            features = [x for x in cur_features if x is not i]

            cv_score = cross_validation_score(self.model, data, features, label, self.metric, fold=5)
            if cv_score > cur_metric:
                cur_metric = cv_score
                removed_feature = i

        if removed_feature == '':
            no_change_value = 1
            print('No more features to be removed based on the step-wise algorithm.')
        else:
            cur_features.remove(removed_feature)
            print('Feature {} to be removed based on the step-wise algorithm.'.format(removed_feature))

        return cur_features, cur_metric, no_change_value


    def select_features(self, data, features, label, method='forward', min_threshold=None):

        if self.sparsify:
            data = data.to_sparse(fill_value=0)

        if min_threshold is not None:
            feature_subset = self.first_criteria(data, features, label, min_threshold)
        else:
            feature_subset = features

        cur_metric = 0
        break_val = 0
        rem_features = feature_subset
        cur_features = []

        print('Starting feature selection')

        while break_val == 0 and len(rem_features) > 0:

            cur_features, rem_features, cur_metric, break_val, new_feature = self.add_new_feature(data, cur_features, rem_features, label, cur_metric)

            if method == 'step-wise' and break_val == 0:
                no_change_value = 0
                while no_change_value == 0 and len(cur_features) > 1:
                    cur_features, cur_metric, no_change_value = self.remove_one_feature(data, cur_features, new_feature, label, cur_metric)

        return cur_features