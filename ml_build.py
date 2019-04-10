import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from data_processing.preprocessing import DataLoader, InteractionDefiner, FeatureSelection

from data_processing.utils import cross_validation_score
from sklearn.linear_model import LogisticRegression

# Load the data
dl = DataLoader()
train, test = dl.load()

# Interaction Definer
Id = InteractionDefiner(['Pclass','SibSp','Parch'],[2,3])
train, test = Id.calculate(train), Id.calculate(test)

# Feature Selection
fs = FeatureSelection()

features = list(train.columns)
features = [col for col in features if col not in ['Survived','Sex','Embarked']]
min_th_feat = fs.first_criteria(train,features,'Survived', 0.6)
print(min_th_feat)