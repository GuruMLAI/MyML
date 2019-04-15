from sklearn.model_selection import KFold
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

class ModelStack:

    def __init__(self, level0_models, level1_model):

        self.level0_models = level0_models
        self.level1_model = level1_model


    def fit0_pred(self, data, features, label, folds=5):

        print('Starting Level 0 fit & predict method on the training data.')
        kf = KFold(n_splits=folds, random_state=25)
        kf.get_n_splits(data)

        trained_level0_models = {x:[] for x in self.level0_models}
        level1_input_vars = ['level0_'+y for y in self.level0_models]

        for i, (train_index, test_index) in enumerate(kf.split(data)):
            x_tr = data.ix[train_index, features]
            y_tr = data.ix[train_index, label]
            x_te = data.ix[test_index, features]

            for j in trained_level0_models:
                model = self.level0_models.get(j)
                model.fit(x_tr, y_tr)

                trained_level0_models.get(j).append(model)

                data.ix[test_index, 'level0_'+j] = model.predict_proba(x_te)[:, 1]

        self.trained_level0_models = trained_level0_models
        self.features = features
        self.label = label
        self.level1_input_vars = level1_input_vars

        corr = data.ix[:,level1_input_vars].corr()

        sns.heatmap(corr, cmap='viridis', vmin=0, vmax=1)
        plt.show()

        print('Level 0 fit & predict method on the training data is complete.')



