import pandas
import numpy

bank = pandas.read_csv("https://raw.githubusercontent.com/byui-cse/cse450-course/master/data/bank.csv")

bank.info()
print(bank.head())