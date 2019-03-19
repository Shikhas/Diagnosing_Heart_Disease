import numpy
import pandas
import matplotlib.pyplot as plt


# 303 patient data
patients_data = pandas.read_csv("heart.csv")

print("First 5 rows are:\n",patients_data.head())
print("Last 5 rows are:\n",patients_data.tail())