from lets_plot import *
import pandas

LetsPlot.setup_html()

bank = pandas.read_csv("https://raw.githubusercontent.com/byui-cse/cse450-course/master/data/bank.csv")

bank.info()