from sklearn.ensemble import AdaBoostClassifier
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import util
import sys

def main():
    opts = util.parse_args()
    X, y = util.data_load(opts.dataset)
    ratio = opts.upsamplet
    n = opts.upsamplen
    if ratio is not None and n is not None:
        print("User can only choose one type of upsample")
        sys.exit()
    elif ratio is not None:
        needed = util.needed_total(X, y, ratio)
        X, y = util.upsample(X, y, needed)
    elif n is not None:
        needed = util.needed_n(X, y, n)
        X, y = util.upsample(X, y, needed)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    X_train, X_test = util.normalize(X_train, X_test)
    clf = AdaBoostClassifier(n_estimators=100, random_state=0)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    conf_mat = confusion_matrix(y_test, predictions)
    print(conf_mat)

if __name__ == '__main__':
    main()
