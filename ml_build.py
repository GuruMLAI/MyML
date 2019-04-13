import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import numpy as np

from sklearn.ensemble import RandomForestClassifier

from data_processing.preprocessing import DataLoader, Encoder, InteractionDefiner, Standardizer
from ml.helpers import FeatureSelector

from sklearn.linear_model import LogisticRegression

run_params = {
    'data_location': '/Users/Guruprasad/Documents/Files/Work/Training/Kaggle/Titanic/data', #'User_Defined' if GUI is to be used to select the folder
    'id_variables': ['PassengerId'],
    'label': ['Survived'],
    'std_variables': ['Fare', 'Age_Imp'],
    'encode_variables': ['Sex','Pclass','Embarked']
}

# Load the data
dl = DataLoader(loc=run_params.get('data_location'))
train, test = dl.load()


# Variable Standardization
Std = Standardizer()
Std.define(train, run_params.get('std_variables'))
train, test = Std.calculate(train), Std.calculate(test)



# Dummy Variable Encoder
if run_params.get('encode_variables') != []:
    ec = Encoder(run_params.get('encode_variables'))
    ec.find_values(train)

    train, test = ec.encode(train), ec.encode(test)


# Interaction Definer
interaction_variables = [col for col in train.columns if col not in run_params.get('id_variables')+run_params.get('label')]
Id = InteractionDefiner(interaction_variables, [2,3], run_params.get('encode_variables'))
train, test = Id.calculate(train), Id.calculate(test)


# Feature Selection
fs = FeatureSelector(model=LogisticRegression(), metric='accuracy')

features = list(train.columns)
base_features = [col for col in features if col not in run_params.get('id_variables')+run_params.get('label')]
#final_features = fs.select_features(train, base_features, run_params.get('label'), 'step-wise', 0.65)
final_features = fs.first_criteria(train, base_features, run_params.get('label'), 0.65)


# Model fit
final_model = RandomForestClassifier()

final_model.fit(np.array(train[final_features]).reshape(-1,len(final_features)),
                np.array(train[run_params.get('label')]).reshape(-1,))

test['Survived'] = [round(value) for value in final_model.predict_proba(test[final_features])[:, 1]]
test[run_params.get('id_variables')+run_params.get('label')].to_csv(run_params.get('data_location')+'/pred.csv')

