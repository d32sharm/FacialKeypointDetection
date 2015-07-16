# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import os

import numpy as np
from pandas.io.parsers import read_csv
from sklearn.utils import shuffle
from matplotlib import pyplot
from sklearn.linear_model import LinearRegression
from sklearn import cross_validation
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
import pickle

FTRAIN = 'data/kaggle-facial-keypoint-detection/training.csv'
FTEST = 'data/kaggle-facial-keypoint-detection/test.csv'


def load(test=False, cols=None):
    """Loads data from FTEST if *test* is True, otherwise from FTRAIN.
    Pass a list of *cols* if you're only interested in a subset of the
    target columns.
    """
    fname = FTEST if test else FTRAIN
    df = read_csv(os.path.expanduser(fname))  # load pandas dataframe

    # The Image column has pixel values separated by space; convert
    # the values to numpy arrays:
    df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' '))

    if cols:  # get a subset of columns
        df = df[list(cols) + ['Image']]

    print(df.count())  # prints the number of values for each column
    df = df.dropna()  # drop all rows that have missing values in them

    X = np.vstack(df['Image'].values) / 255.  # scale pixel values to [0, 1]
    X = X.astype(np.float32) 

    if not test:  # only FTRAIN has any target columns
        y = df[df.columns[:-1]].values
        y = (y - 48) / 48  # scale target coordinates to [-1, 1]
        X, y = shuffle(X, y, random_state=42)  # shuffle train data
        y = y.astype(np.float32)
    else:
        y = None

    return X, y

def plot_sample(x, y, axis):
    img = x.reshape(96, 96)
    axis.imshow(img, cmap='gray')
    axis.scatter(y[0::2] * 48 + 48, y[1::2] * 48 + 48, marker='x', s=10)


X, y = load()

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2, random_state=0)

##############################################################################
'''Linear Regression Model'''
linReg = LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=1)    
linReg.fit(X_train, y_train)   

    
#X_test, _ = load(test=True)
y_pred = linReg.predict(X_test)

fig = pyplot.figure(figsize=(6, 6))
fig.subplots_adjust(
    left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

for i in range(16):
    ax = fig.add_subplot(4, 4, i + 1, xticks=[], yticks=[])
    plot_sample(X_test[i], y_pred[i], ax)
    
pyplot.show()

mean_sq_err = mean_squared_error(y_test, y_pred)
print ('Mean Squared Error = {}').format(mean_sq_err)
print ('Score = {}').format(np.sqrt(mean_sq_err)*48)

with open('linRegression.pickle', 'wb') as f:
    pickle.dump(linReg, f, -1)

##############################################################################
'''SVM SVR'''


