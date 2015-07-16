# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 23:53:52 2015

@author: dhruvsharma
"""
import os

import numpy as np
from pandas.io.parsers import read_csv
from sklearn.utils import shuffle
from matplotlib import pyplot
import pickle

FTRAIN = 'data/kaggle-facial-keypoint-detection/training.csv'
FTEST = 'data/kaggle-facial-keypoint-detection/test.csv'

def load(test=False, cols=None):
    """Loads data from FTEST if *test* is True, otherwise from FTRAIN.
    Pass a list of *cols* if you're only interested in a subset of the
    target columns.
    """
    fname = FTEST if test else FTRAIN
    df = read_csv(os.path.expanduser(fname),nrows=2)  # load pandas dataframe

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
    
X, y = load(test=True)

X_flipped = X.reshape(-1,96,96)[:,::-1]  # simple slice to flip all images
X_flipped = X_flipped.reshape(-1,9216)

linReg = pickle.load( open( "linRegression.pickle", "rb" ) )
y = linReg.predict(X)
y_flipped = linReg.predict(X_flipped)



fig = pyplot.figure(figsize=(6, 3))
ax = fig.add_subplot(1, 2, 1, xticks=[], yticks=[])
plot_sample(X[0], y[0], ax)
ax = fig.add_subplot(1, 2, 2, xticks=[], yticks=[])
plot_sample(X_flipped[0], y[0], ax)
pyplot.show()