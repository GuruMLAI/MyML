from data_processing.preprocessing import DataLoader, Encoder, InteractionDefiner, Standardizer
from ml.helpers import FeatureSelector
from ml.stacking import ModelStack

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression



import warnings
warnings.filterwarnings('ignore')


run_params = {
    'data_location': '/Users/Guruprasad/Documents/Files/Work/Training/Kaggle/Titanic/data', #'User_Defined' if GUI is to be used to select the folder
    'id_variables': ['PassengerId'],
    'label': ['Survived'],
    'std_variables': ['Fare', 'Age'],
    'encode_variables': ['Sex','Embarked', 'Title'],
    'level0_in_level1': []
}

# Load the data
dl = DataLoader(loc=run_params.get('data_location'))
train, test = dl.load()


# Variable Standardization
if run_params.get('std_variables') != []:
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
fs = FeatureSelector(metric='accuracy')

features = list(train.columns)
base_features = [col for col in features if col not in run_params.get('id_variables')+run_params.get('label')]
#final_features = fs.select_features(train, base_features, run_params.get('label'), 'step-wise', 0.65)
final_features = fs.first_criteria(train, base_features, run_params.get('label'), 0.65)


# Start building the model
level0_models = {'rf':RandomForestClassifier(n_estimators=500),
                 'abc':AdaBoostClassifier(n_estimators=500),
                 'gbc':GradientBoostingClassifier(n_estimators=500)
}

MS = ModelStack(level0_models, LogisticRegression(), run_params.get('level0_in_level1'), 'accuracy')

MS.fit_pred(train, final_features, run_params.get('label'), 5)

final = MS.predict(test)

final[run_params.get('id_variables')+run_params.get('label')].to_csv(run_params.get('data_location')+'/final_pred.csv', index = False)

