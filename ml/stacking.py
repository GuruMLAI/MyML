from sklearn.model_selection import KFold
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from ml.utils import cross_validation_score

class ModelStack:

    def __init__(self, level0_models, level1_model, level1_input_vars, metric):

        self.level0_models = level0_models
        self.level1_model = level1_model
        self.metric = metric
        self.level1_input_vars = level1_input_vars



    def fit_pred(self, data, features, label, folds=5):

        print('Starting Level 0 fit & predict method on the training data...')
        kf = KFold(n_splits=folds, random_state=25)
        kf.get_n_splits(data)

        trained_level0_models = {x:[] for x in self.level0_models}
        level1_new_input_vars = ['level0_'+y for y in self.level0_models]
        self.level1_input_vars.extend(level1_new_input_vars)

        for i, (train_index, test_index) in enumerate(kf.split(data)):
            x_tr = data.ix[train_index, features]
            y_tr = data.ix[train_index, label]
            x_te = data.ix[test_index, features]

            for j in trained_level0_models:
                model0 = self.level0_models.get(j)
                model0.fit(x_tr, y_tr)

                trained_level0_models.get(j).append(model0)

                data.ix[test_index, 'level0_'+j] = model0.predict_proba(x_te)[:, 1]

        self.trained_level0_models = trained_level0_models
        self.level0_input_vars = features
        self.label = label

        corr = data.ix[:,level1_new_input_vars].corr()

        sns.heatmap(corr, cmap='viridis', vmin=0, vmax=1)
        plt.show()

        print('Level 0 fit & predict method on the training data is complete.')

        print('Starting to train the level 2 model...')

        model1 = self.level1_model

        model1.fit(data[self.level1_input_vars], data[label])
        cv_score = cross_validation_score(model1, data, self.level1_input_vars, label, self.metric)

        self.trained_level1_model = model1

        print('Completed training the level 1 model.')
        print('The {} score for the final model on the training data is {}'.format(self.metric, cv_score))


    def predict(self,test):

        for i in self.trained_level0_models:
            for fold, j in enumerate(self.trained_level0_models.get(i)):
                var = i+'__'+str(fold)
                test[var] = j.predict_proba(test[self.level0_input_vars])[:,1]

            var_final = 'level0_'+i
            mean_vars = [x for x in list(test.columns) if x.startswith(i+'__')]
            test[var_final] = test[mean_vars].mean(axis=1)

        test[self.label[0]] = [int(round(value)) for value in self.trained_level1_model.predict_proba(test[self.level1_input_vars])[:, 1]]

        return test


