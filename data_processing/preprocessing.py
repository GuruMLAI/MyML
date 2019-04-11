import os, sys
from tkinter import Tk
from tkinter.filedialog import askdirectory

import numpy as np
import pandas as pd
from itertools import combinations

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,RandomForestRegressor,GradientBoostingRegressor
from sklearn.metrics import auc, roc_auc_score, accuracy_score

from data_processing.utils import cross_validation_score

class DataLoader:
    '''
    This can be used for loading the data.
    The user has to choose the folder where the training and the test data are located.
    Also specify the names of the files. The default file format is assumed to be csv.

    The __init__ method defines the data directory and the training and test files.
    The load method does the loading of the data into train and test datasets.

    Alternatively, the user can also directly specify a pathusing the run_params 'Data_Location'
    '''

    def __init__(self,loc = 'User_Defined', train='train.csv', test='test.csv'):

        if loc == 'User_Defined':
            print('Please select the directory that has the data.')
            Tk().withdraw()
            self.data_path = askdirectory()
        elif os.path.exists(loc):
            self.data_path = loc
        else:
            print('The specified path does not exist. Please correct it and try again')
            sys.exit(1)


        self.train_data = os.path.join(self.data_path, train)
        self.test_data = os.path.join(self.data_path, test)

    def load(self):

        return pd.read_csv(self.train_data), pd.read_csv(self.test_data)


class Encoder:
    def __init__(self,features):
        self.features = features

    def find_values(self, train_data):
        feature_values = {}
        for i in self.features:
            feature_values.update({i:list(train_data[i].unique())})
        self.feature_values = feature_values

        print('Unique values of the encoded variables are :\n {}'.format(self.feature_values))
        print('Dummy variables will be created using all but the first value in the list\n')

    def encode(self, data):
        for key, value in self.feature_values.items():
            for i in np.arange(1,len(value)):
                var_name = str(key)+'_'+str(value[i])
                data[var_name] = data[key].apply(lambda x: 1 if x == value[i] else 0)
            data.drop(key, axis=1, inplace=True)

        return data




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


    def first_criteria(self, data, features, label, min_threshold=None):

        if self.sparsify:
            data = data.to_sparse(fill_value=0)

        min_threshold_featres = []

        print('Applying the minimum threshold criteria...\n')
        for i in features:
            cv_score = cross_validation_score(self.model, data, [i], label, self.metric)

            if cv_score > min_threshold:
                min_threshold_featres.append(i)

        print('The following {} variables passed the minimum criteria.\n'.format(len(min_threshold_featres)))
        print('{}'.format(min_threshold_featres))

        return min_threshold_featres


    def select_features(self, data, features, label, min_threshold=None):

        if self.sparsify:
            data = data.to_sparse(fill_value=0)

        if min_threshold is not None:
            feature_subset = self.first_criteria(data, features, label, min_threshold)
        else:
            feature_subset = features


        feature_comb = []
        for i in np.arange(1,len(feature_subset)+1):
            for x in combinations(np.arange(len(feature_subset)),i):
                feature_comb.append(x)

        final_metric = 0
        final_features = []

        print('Starting feature selection')
        for i in feature_comb:
            f_list = [feature_subset[k] for k in list(i)]
            cv_score = cross_validation_score(self.model, data, f_list, label, self.metric, fold=5)
            if cv_score > final_metric:
                final_metric = cv_score
                final_features = f_list
                print('Updating the feature list to {} with {} of {}'.format(f_list, self.metric, final_metric))

        return final_features
