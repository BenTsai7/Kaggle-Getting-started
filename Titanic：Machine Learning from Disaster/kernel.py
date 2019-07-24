# data analysis and wrangling
import pandas as pd
import numpy as np
import csv
import random as rnd

# visualization
import matplotlib.pyplot as plt

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier


#分析数据的过程中进行清洗转换
def analysis_and_wrangle_data():
    train_df = pd.read_csv('data/train.csv')
    test_df = pd.read_csv('data/test.csv')
    combine = [train_df, test_df]
    print(train_df.columns.values)
    #print(train_df.head())# 表头尾预览
    #print(train_df.tail())
    print(train_df.info())
    print(test_df.info())
    # Cabin > Age > Embarked features contain a number of null values in that order for the training dataset.
    # Cabin > Age are incomplete in case of test dataset.'''
    # 通过这一步发现Cabin，Age和Embarked是有缺失的，并且Cabin影响不大，可以扔掉
    '''通过分析数据扔掉关联性不大的数据'''
    train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)
    test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)
    combine = [train_df, test_df]
    print('_'*50)
    print(train_df.describe(include=['O']))
    print('_' * 50)
    print(train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False))
    print(train_df[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False))
    print(train_df[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False))
    print(train_df[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived',ascending=False))
    # 分析name属性
    print('_' * 50)
    for dataset in combine:
        dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    print(pd.crosstab(train_df['Title'], train_df['Sex']))
    print('_' * 50)
    # 对Miss,Mr等进行分析
    for dataset in combine:
        dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess', 'Capt', 'Col',
                                                     'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
        dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
        dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
        dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    print('_' * 50)
    # 将原有姓名提取出以下的Title属性，添加入原有数据集
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    for dataset in combine:
        dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    for dataset in combine:
        dataset['Title'] = dataset['Title'].map(title_mapping)
        dataset['Title'] = dataset['Title'].fillna(0)
    print(train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean())
    print('_' * 50)
    # 扔掉Name 和 PassageId属性
    train_df = train_df.drop(['Name', 'PassengerId'], axis=1)
    test_df = test_df.drop(['Name'], axis=1)
    combine = [train_df, test_df]
    # 性别转换为0，1
    for dataset in combine:
        dataset['Sex'] = dataset['Sex'].map({'female': 1, 'male': 0}).astype(int)
    print(train_df.head())
    # age转换为整数
    guess_ages = np.zeros((2, 3))
    for dataset in combine:
        for i in range(0, 2):
            for j in range(0, 3):
                guess_df = dataset[(dataset['Sex'] == i) & \
                                   (dataset['Pclass'] == j + 1)]['Age'].dropna()
                age_guess = guess_df.median()
                # Convert random age float to nearest .5 age
                guess_ages[i, j] = int(age_guess / 0.5 + 0.5) * 0.5
        for i in range(0, 2):
            for j in range(0, 3):
                dataset.loc[(dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j + 1), \
                            'Age'] = guess_ages[i, j]
        dataset['Age'] = dataset['Age'].astype(int)
    # 分析age
    train_df['AgeBand'] = pd.cut(train_df['Age'], 5)
    print(train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand',
                                                                                              ascending=True))
    #age按等级切分
    for dataset in combine:
        dataset.loc[dataset['Age'] <= 16, 'Age'] = 0
        dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
        dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
        dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
        dataset.loc[dataset['Age'] > 64, 'Age'] = 4
    print(train_df.head())
    #移除Ageband
    train_df = train_df.drop(['AgeBand'], axis=1)
    combine = [train_df, test_df]
    train_df.head()
    print(train_df.head())
    # 分析SibSp和Parch
    for dataset in combine:
        dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
    print(train_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived',
                                                                                                    ascending=False))
    for dataset in combine:
        dataset['IsAlone'] = 0
        dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1
    print(train_df[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean())
    train_df = train_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
    test_df = test_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
    combine = [train_df, test_df]
    for dataset in combine:
        dataset['Age*Class'] = dataset.Age * dataset.Pclass
    print(train_df.loc[:, ['Age*Class', 'Age', 'Pclass']].head(10))
    #最频繁出现的缺失补上并量化
    freq_port = train_df.Embarked.dropna().mode()[0]
    for dataset in combine:
        dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)
    for dataset in combine:
        dataset['Embarked'] = dataset['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)
    test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)
    train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)
    train_df[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand',
                                                                                                ascending=True)
    for dataset in combine:
        dataset.loc[dataset['Fare'] <= 7.91, 'Fare'] = 0
        dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
        dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare'] = 2
        dataset.loc[dataset['Fare'] > 31, 'Fare'] = 3
        dataset['Fare'] = dataset['Fare'].astype(int)

    train_df = train_df.drop(['FareBand'], axis=1)
    combine = [train_df, test_df]
    print(train_df.head(10))
    X_train = train_df.drop("Survived", axis=1)
    Y_train = train_df["Survived"]
    X_test = test_df.drop("PassengerId", axis=1).copy()
    return X_train, Y_train, X_test


def predict(X_train, Y_train, X_test):
    random_forest = RandomForestClassifier(n_estimators=100)
    random_forest.fit(X_train, Y_train)
    Y_pred = random_forest.predict(X_test)
    random_forest.score(X_train, Y_train)
    acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
    print(acc_random_forest)
    return  Y_pred


def write2csv(Y_pred, X_train):
    file = open('data/gender_submission.csv', 'w', newline='',)
    writer = csv.writer(file)
    writer.writerow(('PassengerId', 'Survived'))
    row = len(X_train) + 1
    for index in range(len(Y_pred)):
        writer.writerow((row, Y_pred[index]))
        row += 1
    file.close()

X_train, Y_train, X_test = analysis_and_wrangle_data()
Y_pred = predict(X_train, Y_train, X_test)
write2csv(Y_pred, X_train)
