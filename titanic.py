#coding:utf-8
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# 将 titanic 数据读取到内存中，以 pandas 的 DataFrame 储存
titanic_df = pd.read_csv("./train.csv")
test_df    = pd.read_csv("./test.csv")



# 对数据进行初步处理，因为乘客编号，姓名，与船票编号与是否生存是无关数据
# 所以我们去除这三个数据列
titanic_df = titanic_df.drop(['PassengerId','Name','Ticket'], axis=1)
test_df    = test_df.drop(['Name','Ticket'], axis=1)
test_df.head()

X = titanic_df['Age'].fillna('0')
y = titanic_df['Survived']

print y.shape