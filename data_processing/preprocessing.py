import os
from tkinter import Tk
from tkinter.filedialog import askdirectory

import numpy as np
import pandas as pd
from itertools import combinations

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,RandomForestRegressor,GradientBoostingRegressor
from sklearn.metrics import auc, roc_auc_score, accuracy_score

class DataLoader:
    '''
    This can be used for loading the data.
    The user has to choose the folder where the training and the test dat aare located.
    Also specify the names of the files. The default file format is assumed to be csv.

    The __init__ method defines the data directory and the training and test files.
    The load method does the loading of the data into train and test datasets.
    '''

    def __init__(self, train='train.csv', test='test.csv'):

        print('Please select the directory that has the data.')
        Tk().withdraw()
        self.data_path = askdirectory()

        self.train_data = os.path.join(self.data_path, train)
        self.test_data = os.path.join(self.data_path, test)

    def load(self):

        return pd.read_csv(self.train_data), pd.read_csv(self.test_data)


class InteractionDefiner:
    '''
    This class defines the Interactions in the dataset between a defined set of input variables.
    The __init__ method defines the iterator which generates all the possible combinations of the variables
    that have been defined in features

    The type of interactions that are intended to be explored (e.g 2-way, 3-way etc.
     can be defined in types as a list
    '''

    def __init__(self, features, types):

        self.features = features
        self.types = types

        self.iterator = []
        for i in self.types:
            for x in combinations(np.arange(len(self.features)),i):
                self.iterator.append(x)

        print(self.iterator)


    def calculate(self, data):

        for i in self.iterator:

            f_list = [self.features[k] for k in list(i)]
            f_name = '_'.join(f_list)
            data[f_name] = data[f_list].prod(axis=1)

        return data


class FeatureSelection:

    def __init__(self,model=LogisticRegression, mertic='auc', sparsify=True):
        self.model = model
        self.metric = mertic
        self.sparsify = sparsify

    def find(self, data, features, label, min_threshold=None,):
        if self.sparsify:
            data = data.to_sparse(fill_value=0)

        X = data[features]
        y = data[label]

        min_threshold_featres = []
        if min_threshold is not None:
            print('Applying the minimum threshold criteria...')
            for i in features:
                th_model = self.model
                th_model.fit(X[i],y)
                var_metric = roc_auc_score(y,th_model.predict_proba(X[i]))
                if var_metric > min_threshold:
                    min_threshold_featres.append(i)






