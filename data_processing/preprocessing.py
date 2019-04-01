import os
from tkinter import Tk
from tkinter.filedialog import askdirectory

import numpy as np
import pandas as pd
from itertools import combinations

class DataLoader:

    def __init__(self, train='train.csv', test='test.csv'):

        print('Please select the directory that has the data.')
        Tk().withdraw()
        self.data_path = askdirectory()

        self.train_data = os.path.join(self.data_path, train)
        self.test_data = os.path.join(self.data_path, test)

    def load(self):

        return pd.read_csv(self.train_data), pd.read_csv(self.test_data)


class InteractionDefiner:

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


