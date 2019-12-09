import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import optparse
import sys
import matplotlib.pyplot as plt



def parse_args():
    """Parse command line arguments (train and test arff files)."""
    parser = optparse.OptionParser(description='run ensemble method')

    parser.add_option('-d', '--dataset', type='string', help='path to' +\
        ' data file')


    (opts, args) = parser.parse_args()

    mandatories = ['dataset',]
    for m in mandatories:
        if not opts.__dict__[m]:
            print('mandatory option ' + m + ' is missing\n')
            parser.print_help()
            sys.exit()

    return opts

def data_load(filename):
    data = pd.read_csv(filename).to_numpy(dtype=np.float32)
    # Get rid of time feature to see what happens`
    X, y = data[:, 1:-1], data[:, -1]
    X, y = shuffle(X, y)
    return X, y

def normalize(X_train, X_test):
    mean_pixel = X_train.mean(axis=(0, 1), keepdims=True)
    std_pixel = X_train.std(axis=(0, 1), keepdims=True)
    X_train = X_train - mean_pixel
    X_train = X_train / std_pixel
    X_test = X_test - mean_pixel
    X_test = X_test / std_pixel
    return X_train, X_test

data_load("temp.csv")
