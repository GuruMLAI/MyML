from data_processing.preprocessing import DataLoader, InteractionDefiner

# Load the data
dl = DataLoader()
train, test = dl.load()
print(train)

# Interaction Definer
Id = InteractionDefiner(['Pclass','SibSp','Parch','Survived'],[2,3])
print(Id.calculate(train))


