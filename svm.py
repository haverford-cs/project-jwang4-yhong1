'''
Initialize SVM 
12/8 2019
'''
import util
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report


def main():
    opts = util.parse_args()
    X, y = util.data_load(opts.dataset)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    model=SVC()
    #old grid
    # param_grid = {"C": [1, 10, 100, 1000], "gamma": [1e-4,1e-3,1e-2,1e-1,1]}
    #new grid
    param_grid = {"C": [1, 10, 100, 1000], "gamma": [1e-4,1e-3,1e-2,1e-1]}
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        grid=GridSearchCV(model, param_grid,cv=3, verbose=4,iid=False)
        grid.fit(X_train,y_train)
        grid_predictions=grid.predict(X_test)
        cm=confusion_matrix(y_test,grid_predictions)
        report=classification_report(y_test,grid_predictions)
        
        ''' 
        without doing the gridsearch just using the default params - bad results
        model.fit(X_train,y_train)
        pred=model.predict(X_test)
        cm=confusion_matrix(y_test,pred)
        report=classification_report(y_test,pred)
        '''

        print(cm)
        print(report)
        print("Best_Param_Estimator:", grid.best_estimator_)


if __name__ == '__main__':
    main()