import os
from tkinter import Tk
from tkinter.filedialog import askdirectory

import pandas as pd

class DataLoader:

    def __init__(self, train='train.csv', test='test.csv'):
        print('Please select the directory that has the data.')
        Tk().withdraw()
        self.data_path = askdirectory()

        self.train_data = os.path.join(self.data_path, train)
        self.test_data = os.path.join(self.data_path, test)

    def load(self):

        return pd.read_csv(self.train_data), pd.read_csv(self.test_data)

