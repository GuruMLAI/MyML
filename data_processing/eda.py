# Some aspects of this eda have been taken from available public kernels mainly: Ansiotropic (https://www.kaggle.com/arthurtok/introduction-to-ensembling-stacking-in-python)

import numpy as np
import pandas as pd
import re
path = '/Users/Guruprasad/Documents/Files/Work/Training/Kaggle/Titanic/data'

train = pd.read_csv(path+'/titanic_train.csv')
test = pd.read_csv(path+'/titanic_test.csv')

print('Columns not in train: {}'.format([x for x in test.columns if x not in train.columns]))
print('Columns not in test: {}'.format([x for x in train.columns if x not in test.columns]))

train['identifier'] = 'train'
test['identifier'] = 'test'

full_data = pd.concat((train, test))

full_data.info()

# Gives the length of the name
train['Name_length'] = train['Name'].apply(len)
test['Name_length'] = test['Name'].apply(len)
# Feature that tells whether a passenger had a cabin on the Titanic
train['Has_Cabin'] = train["Cabin"].apply(lambda x: 0 if type(x) == float else 1)
test['Has_Cabin'] = test["Cabin"].apply(lambda x: 0 if type(x) == float else 1)

# Create new feature FamilySize as a combination of SibSp and Parch
full_data['FamilySize'] = full_data['SibSp'] + full_data['Parch'] + 1

# Create new feature IsAlone from FamilySize
full_data['IsAlone'] = full_data['FamilySize'].apply(lambda x: 1 if x==1 else 0)

# Remove all NULLS in the Embarked column
full_data['Embarked'] = full_data['Embarked'].fillna('S')

# Remove all NULLS in the Fare column and create a new feature CategoricalFare
full_data['Fare'] = full_data['Fare'].fillna(train['Fare'].median())
# train['CategoricalFare'] = pd.qcut(train['Fare'], 4)

# Create a New feature CategoricalAge
age_avg = full_data['Age'].mean()
age_std = full_data['Age'].std()
age_null_count = full_data['Age'].isnull().sum()
age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
full_data['Age'][np.isnan(full_data['Age'])] = age_null_random_list
full_data['Age'] = full_data['Age'].astype(int)
# train['CategoricalAge'] = pd.cut(train['Age'], 5)


# Define function to extract titles from passenger names
def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""


# Create a new feature Title, containing the titles of passenger names
full_data['Title'] = full_data['Name'].apply(get_title)

# Group all non-common titles into one single grouping "Rare"
full_data['Title'] = full_data['Title'].replace(
        ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

full_data['Title'] = full_data['Title'].replace('Mlle', 'Miss')
full_data['Title'] = full_data['Title'].replace('Ms', 'Miss')
full_data['Title'] = full_data['Title'].replace('Mme', 'Mrs')

print(full_data.columns)

cols_train = ['Age', 'Embarked', 'Fare', 'Parch', 'PassengerId',
       'Pclass', 'Sex', 'SibSp', 'Survived',
       'FamilySize', 'IsAlone', 'Title']

cols_test = ['Age', 'Embarked', 'Fare', 'Parch', 'PassengerId',
       'Pclass', 'Sex', 'SibSp',
       'FamilySize', 'IsAlone', 'Title']

full_data[full_data['identifier'] == 'train'][cols_train].to_csv(path+'/train.csv', index=False)

full_data[full_data['identifier'] == 'test'][cols_test].to_csv(path+'/test.csv', index=False)




