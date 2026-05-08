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
mini_holdout = pandas.read_csv("https://raw.githubusercontent.com/byui-cse/cse450-course/master/data/bank_holdout_test_mini.csv")

bank.info()

#Actually making and training the model 

Y = bank['y'].map({'yes': 1, 'no': 0})

bank = bank.drop(columns = ['y', 'contact', 'month', 'campaign', 'pdays'])

#"basic.4y","basic.6y","basic.9y","high.school","illiterate","professional.course","university.degree","unknown"

bank['education'] = bank['education'].map({
    'unknown': 0,
    'illiterate': -1,
    'basic.4y': 1,
    'basic.6y': 2,
    'basic.9y': 3,
    'high.school': 4,
    'professional.course': 5,
    'university.degree': 6
})

bank['default'] = bank['default'].map({
    'no': -1,
    'unknown': 0,
    'yes': 1
})

bank['housing'] = bank['housing'].map({
    'no': -1,
    'unknown': 0,
    'yes': 1
})

bank['day_of_week'] = bank['day_of_week'].map({
    'mon': 1,
    'tue': 2,
    'wed': 3,
    'thu': 4,
    'fri': 5
})

bank['poutcome'] = bank['poutcome'].map({
    'nonexistent': 0,
    'failure': -1,
    'success': 1
})

bank['loan'] = bank['loan'].map({
    'no': -1,
    'unknown': 0,
    'yes': 1
})

bank_ohe = pandas.get_dummies(bank, drop_first = True)

X = bank_ohe

X_Train, X_Test, Y_Train, Y_Test = train_test_split(
    X, Y, test_size = 0.3, random_state = 69
)

model = RandomForestClassifier(n_estimators= 200, random_state = 69)

model.fit(X_Train, Y_Train)

pred = model.predict(X_Test)
print(model.score(X_Test, Y_Test))
print(classification_report(Y_Test, pred))


# This is all for the mini holdout thing 

mini_bank = mini_holdout.copy()

mini_bank = mini_bank.drop(columns = ['contact', 'month', 'campaign', 'pdays'])

mini_bank['education'] = mini_bank['education'].map({
    'unknown': 0,
    'illiterate': -1,
    'basic.4y': 1,
    'basic.6y': 2,
    'basic.9y': 3,
    'high.school': 4,
    'professional.course': 5,
    'university.degree': 6
})

mini_bank['default'] = mini_bank['default'].map({
    'no': -1,
    'unknown': 0,
    'yes': 1
})

mini_bank['housing'] = mini_bank['housing'].map({
    'no': -1,
    'unknown': 0,
    'yes': 1
})

mini_bank['day_of_week'] = mini_bank['day_of_week'].map({
    'mon': 1,
    'tue': 2,
    'wed': 3,
    'thu': 4,
    'fri': 5
})

mini_bank['poutcome'] = mini_bank['poutcome'].map({
    'nonexistent': 0,
    'failure': -1,
    'success': 1
})

mini_bank['loan'] = mini_bank['loan'].map({
    'no': -1,
    'unknown': 0,
    'yes': 1
})

X2 = pandas.get_dummies(mini_bank, drop_first = True)

X2.info()

pred2 = model.predict(X2)

print(pred2)

pandas.DataFrame(pred2).to_csv('mini-predictions.csv', index=False)