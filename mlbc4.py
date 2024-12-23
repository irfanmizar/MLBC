# -*- coding: utf-8 -*-
"""MLBC4.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1Fbaro1pa29u7XiMvleIwtHECT90OLQnQ
"""

import pandas as pd
from sklearn.datasets import load_wine

data = load_wine()
df = pd.DataFrame(data.data, columns=data.feature_names)
df["target"] = data.target
#df.isna().sum()
df.dropna(inplace=True)

#data.target_names
df["target"].value_counts()

import seaborn as sns

sns.boxplot(df.iloc[:, :4])

from sklearn.model_selection import train_test_split

X = df.drop(columns=["target"])
Y = df["target"]
x_train, x_test, y_train, y_test = train_test_split(X, Y)

y_train.shape

from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()
model.fit(x_train, y_train)

plot_tree(model)

model.score(x_test, y_test)