import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from data_processing.preprocessing import DataLoader, Encoder, InteractionDefiner
from ml.helpers import FeatureSelector

run_params = {
    'Data_Location': '/Users/Guruprasad/Documents/Files/Work/Training/Kaggle/Titanic/data', #'User_Defined' if GUI is to be used to select the folder
    'ID_variables': ['PassengerId'],
    'label': ['Survived'],
    'encode_variables': ['Sex','Pclass','Embarked']
}

# Load the data
dl = DataLoader(loc=run_params.get('Data_Location'))
train, test = dl.load()


# Dummy Variable Encoder
ec = Encoder(run_params.get('encode_variables'))
ec.find_values(train)

train = ec.encode(train)
test = ec.encode(test)


# Interaction Definer
interaction_variables = [col for col in train.columns if col not in run_params.get('ID_variables')+run_params.get('label')]
Id = InteractionDefiner(interaction_variables, [2,3], run_params.get('encode_variables'))
train, test = Id.calculate(train), Id.calculate(test)


# Feature Selection
fs = FeatureSelector()

features = list(train.columns)
base_features = [col for col in features if col not in run_params.get('ID_variables')+run_params.get('label')]
final_features = fs.select_features(train, base_features, run_params.get('label'), 'step-wise', 0.6)