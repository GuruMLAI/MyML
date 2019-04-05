from data_processing.preprocessing import DataLoader, InteractionDefiner

from data_processing.utils import cross_validation_score

# Load the data
dl = DataLoader()
train, test = dl.load()
#print(train)

cross_validation_score(train,['Pclass','SibSp','Parch'],'Survived', 3 )

# Interaction Definer
#Id = InteractionDefiner(['Pclass','SibSp','Parch','Survived'],[2,3])
#print(Id.calculate(train))


