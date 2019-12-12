'''
Initialize SVM 
12/8 2019
'''
import util
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

#running in lab machine 
#python3 svm.py -d /homes/yhong1/cs360/Final_Project/data/creditcard.csv
def main():
    opts = util.parse_args()
    X, y = util.data_load(opts.dataset)
    #set upsample ratios or n
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

    #create & train models
    model=SVC()
    param_grid = {"C": [1, 10, 100, 1000], "gamma": [1e-4,1e-3,1e-2,1e-1,1]}
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)
    X_train, X_test = util.normalize(X_train, X_test)

    grid=GridSearchCV(model, param_grid,cv=3, verbose=4,iid=False)
    grid.fit(X_train,y_train)
    grid_predictions=grid.predict(X_test)
    cm=confusion_matrix(y_test,grid_predictions)
    report=classification_report(y_test,grid_predictions)

    print(cm)
    print(report)
    print("Best_Param_Estimator:", grid.best_estimator_)


if __name__ == '__main__':
    main()