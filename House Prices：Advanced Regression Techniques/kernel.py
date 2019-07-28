# data analysis and wrangling
import pandas as pd
import numpy as np
import csv
import random as rnd

# visualization
import matplotlib.pyplot as plt

# machine learning
from sklearn.ensemble import GradientBoostingRegressor


#分析数据的过程中进行清洗转换
def analysis_and_wrangle_data():
    train_df = pd.read_csv('data/train.csv')
    test_df = pd.read_csv('data/test.csv')
    combine = [train_df, test_df]
    print(train_df.columns.values)
    print(train_df.info())
    print(test_df.info())
    print('*' * 50)
    # 分析缺失值
    nullarr = train_df.isnull().sum()
    print(nullarr[nullarr > 0].sort_values(ascending=False))
    # 去掉缺失值过高的特征
    train_df = train_df.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu'], axis=1)
    test_df = test_df.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu'], axis=1)
    cols = [ "GarageQual", "GarageCond", "GarageFinish","GarageYrBlt", "GarageType",
             "BsmtExposure", "BsmtCond", "BsmtQual", "BsmtFinType2", "BsmtFinType1",
             "MasVnrType"]
    for col in cols:
        train_df[col].fillna("None", inplace=True)
        test_df[col].fillna("None", inplace=True)
    cols = ["MasVnrArea", "BsmtUnfSF", "TotalBsmtSF", "GarageCars", "BsmtFinSF2", "BsmtFinSF1", "GarageArea"]
    for col in cols:
        train_df[col].fillna(0, inplace=True)
        test_df[col].fillna(0, inplace=True)
    x = train_df.loc[train_df["LotFrontage"].notnull(), "LotArea"]
    y = train_df.loc[train_df["LotFrontage"].notnull(), "LotFrontage"]
    t = (x <= 25000) & (y <= 150)
    p = np.polyfit(x[t], y[t], 1)
    train_df.loc[train_df['LotFrontage'].isnull(), 'LotFrontage'] = \
        np.polyval(p, train_df.loc[train_df['LotFrontage'].isnull(), 'LotArea'])
    x = test_df.loc[train_df["LotFrontage"].notnull(), "LotArea"]
    y = test_df.loc[train_df["LotFrontage"].notnull(), "LotFrontage"]
    t = (x <= 25000) & (y <= 150)
    p = np.polyfit(x[t], y[t], 1)
    test_df.loc[test_df['LotFrontage'].isnull(), 'LotFrontage'] = \
        np.polyval(p, test_df.loc[test_df['LotFrontage'].isnull(), 'LotArea'])
    train_df['Electrical'] = train_df['Electrical'].fillna(train_df['Electrical'].mode()[0])
    test_df['Electrical'] = test_df['Electrical'].fillna(test_df['Electrical'].mode()[0])
    test_df["MSZoning"] = test_df.groupby("MSSubClass")["MSZoning"].transform(lambda x: x.fillna(x.mode()[0]))
    test_df["Functional"] = test_df["Functional"].fillna(test_df['Functional'].mode()[0])
    test_df["Utilities"] = test_df["Utilities"].fillna(test_df['Utilities'].mode()[0])
    test_df["KitchenQual"] = test_df["KitchenQual"].fillna(test_df['KitchenQual'].mode()[0])
    test_df["SaleType"] = test_df["SaleType"].fillna(test_df["SaleType"].mode()[0])
    test_df["BsmtHalfBath"].fillna(0, inplace=True)
    test_df["BsmtFullBath"].fillna(0, inplace=True)
    test_df['Exterior1st'] = test_df['Exterior1st'].fillna(test_df['Exterior1st'].mode()[0])
    test_df['Exterior2nd'] = test_df['Exterior2nd'].fillna(test_df['Exterior2nd'].mode()[0])
    print('-' * 50)
    df = [train_df, test_df]
    for dateset in df:
        cols = dateset.select_dtypes(exclude=[np.number]).columns
        ordinalList = ['ExterQual', 'ExterCond', 'GarageQual', 'GarageCond',
                       'KitchenQual', 'HeatingQC', 'BsmtQual', 'BsmtCond']
        ordinalmap = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0}
        for c in ordinalList:
            dateset[c] = dateset[c].map(ordinalmap)
        dateset['BsmtExposure'] = dateset['BsmtExposure'].map({'None': 0, 'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4})
        dateset['BsmtFinType1'] = dateset['BsmtFinType1'].map(
            {'None': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6})
        dateset['BsmtFinType2'] = dateset['BsmtFinType2'].map(
            {'None': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6})
        dateset['Functional'] = dateset['Functional'].map(
            {'Maj2': 1, 'Sev': 2, 'Min2': 3, 'Min1': 4, 'Maj1': 5, 'Mod': 6, 'Typ': 7})
        dateset['GarageFinish'] = dateset['GarageFinish'].map({'None': 0, 'Unf': 1, 'RFn': 2, 'Fin': 3})
    combine = pd.concat((train_df, test_df), axis=0)
    convert(df, combine)
    cols = train_df.select_dtypes(exclude=[np.number]).columns
    print(cols)
    return df[0].drop("SalePrice", axis=1), df[0]["SalePrice"] ,df[1]


def str2num(col):
    dict = {}
    unique_str = np.unique(col)
    for num,value in enumerate(unique_str,start = 1):
        dict[value] = num
    return dict


def convert(df, combine):
    dict = {}
    features = combine.dtypes[combine.dtypes == "object"].index
    for obj in features:
        dict[obj] = str2num(combine[obj].astype('str'))
    for dateset in df:
        for i in dict.keys():
            dateset[i] = dateset[i].apply(str).apply(lambda x: dict[i].get(x, np.nan))


def predict(X_train, Y_train, X_test):
    gbr = GradientBoostingRegressor()
    gbr.fit(X_train, Y_train)
    Y_pred = gbr.predict(X_test)
    print(gbr.score(X_train, Y_train))
    return  Y_pred


def write2csv(Y_pred, X_train):
    file = open('data/gender_submission.csv', 'w', newline='',)
    writer = csv.writer(file)
    writer.writerow(('Id', 'SalePrice'))
    row = len(X_train) + 1
    for index in range(len(Y_pred)):
        writer.writerow((row, Y_pred[index]))
        row += 1
    file.close()


X_train, Y_train, X_test = analysis_and_wrangle_data()
Y_pred = predict(X_train, Y_train, X_test)
write2csv(Y_pred, X_train)
