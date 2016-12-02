#coding:utf-8
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, cross_val_score
from sklearn import preprocessing
from sklearn.svm import SVC
# 将 titanic 数据读取到内存中，以 pandas 的 DataFrame 储存
titanic_df = pd.read_csv("./data/train.csv")
test_df    = pd.read_csv("./data/test.csv")



# 对数据进行初步处理，因为乘客编号，姓名，与船票编号与是否生存是无关数据
# 所以我们去除这三个数据列
titanic_df = titanic_df.drop(['PassengerId','Name','Ticket'], axis=1)
# 对 Sex Cabin Embarked 进行 One-hot 编码
preEncoder = preprocessing.LabelEncoder()
titanic_df['Sex'] = preEncoder.fit_transform(titanic_df['Sex'])
titanic_df['Cabin'] = preEncoder.fit_transform(titanic_df['Cabin'])
titanic_df['Embarked'] = preEncoder.fit_transform(titanic_df['Embarked'])

X = titanic_df.fillna('0')

train = X
label = X['Survived']
foldNum = -90
kfold = KFold(n_splits=10)
clf = SVC()
clf.fit(train[:foldNum], label[:foldNum])
# print cross_val_score(clf, train, label, cv=kfold, n_jobs=-1)
print clf.score(train[foldNum:], label[foldNum:])
