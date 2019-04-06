from data_processing.preprocessing import DataLoader, InteractionDefiner

from data_processing.utils import cross_validation_score
from sklearn.linear_model import LogisticRegression

# Load the data
dl = DataLoader()
train, test = dl.load()
#print(train)

model, metric = cross_validation_score(LogisticRegression, train,['Pclass','SibSp','Parch'],'Survived', 'auc', 3 )

# Interaction Definer
#Id = InteractionDefiner(['Pclass','SibSp','Parch','Survived'],[2,3])
#print(Id.calculate(train))


