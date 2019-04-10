import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from data_processing.preprocessing import DataLoader, Encoder, InteractionDefiner, FeatureSelection

from data_processing.utils import cross_validation_score
from sklearn.linear_model import LogisticRegression

run_params = {
    'ID_variables': ['PassengerId'],
    'label': ['Survived'],
    'encode_variables': ['Sex','Pclass','Embarked']
}

# Load the data
dl = DataLoader()
train, test = dl.load()

# Encoder
ec = Encoder(run_params.get('encode_variables'))
ec.find_values(train)

train = ec.encode(train)
test = ec.encode(test)

# Interaction Definer
interaction_variables = [col for col in train.columns if col not in run_params.get('ID_variables')+run_params.get('label')]
Id = InteractionDefiner(interaction_variables,[2,3])
train, test = Id.calculate(train), Id.calculate(test)
print(train.columns)

'''
# Feature Selection
fs = FeatureSelection()

features = list(train.columns)
features = [col for col in features if col not in ['Survived','Sex','Embarked']]
min_th_feat = fs.first_criteria(train, features, run_params.get('label'), 0.6)
print(min_th_feat)

'''

