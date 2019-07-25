import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import csv


def normalize(x):
    x = int(x)
    if x != 0:
        x = 1
    return x


def get_data():
    train_df = pd.read_csv('data/train.csv')
    test_df = pd.read_csv('data/test.csv')
    train_x = train_df.drop('label', axis=1)
    train_y = train_df['label']
    train_x.applymap(normalize)
    return train_x, train_y, test_df


def predict(train_x, train_y, test_x):
    clf = RandomForestClassifier(n_estimators=200,warm_start = True)
    clf.fit(train_x, train_y)
    y_pred = clf.predict(test_x)
    acc_svm = round(clf.score(train_x, train_y) * 100, 2)
    print(acc_svm)
    return y_pred


def write2csv(Y_pred):
    file = open('data/gender_submission.csv', 'w', newline='',)
    writer = csv.writer(file)
    writer.writerow(('ImageId', 'Label'))
    row = 1
    for index in range(len(Y_pred)):
        writer.writerow((row, Y_pred[index]))
        row += 1
    file.close()


train_x, train_y, test_x = get_data()
y_pred = predict(train_x, train_y, test_x)
write2csv(y_pred)
