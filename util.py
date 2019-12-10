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
    parser.add_option('-u', '--upsample', type='float', help='ratio of upsample' +\
            'examples with true label')

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

def upsample(X, y, ratio)
    count_true = np.sum(y)
    needed = int((ratio*count-count_true) / (1-ratio))
    tx= extract_true(X, y)
    result_X, result_y = X, y
    for i in range(needed):
        idx = int(needed * np.random.random_sample())
        result_X = np.append(result_X, [X[idx]], axis=0)
        result_y = np.append(result_y, 1)
    assert(len(result_X) == len(result_y))
    return result_X, result_y

def extract_true(X, y):
    result_x= []
    for i in range(len(y)):
        if y[i] == 1:
            result_x.append(X[i])
    return result_x

def get_roc_curve(mats, model_name):
    xs, ys = [], []
    for confusion_matrix in mats:
        false_positive = confusion_matrix[0][1] / (confusion_matrix[0][0] + confusion_matrix[0][1])
        true_positive = confusion_matrix[1][1] / (confusion_matrix[1][0] + confusion_matrix[1][1])
        xs.append(false_positive)
        ys.append(true_positive)
    plt.plot(xs, ys, 'b')
    plt.title("Roc curve for Credit Card Fraud Detection using, " + model_name)
    plt.xlabel("false positive rate")
    plt.ylabel("true positive rate")
    plt.show()
