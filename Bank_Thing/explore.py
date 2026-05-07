import pandas
import numpy
import sklearn
from lets_plot import *
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

bank = pandas.read_csv("https://raw.githubusercontent.com/byui-cse/cse450-course/master/data/bank.csv")

bank.info()

Y = bank['y'].map({'yes': 1, 'no': 0})

features = bank[['age', 'marital', 'education', 'housing', 'previous', 'poutcome']].copy()

X = pandas.get_dummies(features)

X_Train, X_Test, Y_Train, Y_Test = train_test_split(
    X, Y, test_size = 0.3, random_state = 69
)

model = RandomForestClassifier(n_estimators= 50, random_state = 69)

model.fit(X_Train, Y_Train)
pred = model.predict(X_Test)
print(model.score(X_Test, Y_Test))
print(classification_report(Y_Test, pred))