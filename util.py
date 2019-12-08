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
    return X[:100000], y[:100000]



data_load("temp.csv")
