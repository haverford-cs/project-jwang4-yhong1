'''
Initialize SVM 
12/8 2019
'''
import util
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report, plot_roc_curve
from sklearn.model_selection import train_test_split
import sys
from sklearn.metrics import average_precision_score, precision_recall_curve, plot_precision_recall_curve
import matplotlib.pyplot as plt

#running in lab machine 
#python3 svm.py -d /homes/yhong1/cs360/Final_Project/data/creditcard.csv -n 10
#python3 svm.py -d /homes/yhong1/cs360/Final_Project/data/creditcard.csv -s 1 -n 10
#python3 adaboost.py -d /homes/yhong1/cs360/Final_Project/data/creditcard.csv -t 0.5 -n 10
#python3 fc.py -d /homes/yhong1/cs360/Final_Project/data/creditcard.csv -s 1 -n 10
#python3 run_pipeline.py -d /homes/yhong1/cs360/Final_Project/data/creditcard.csv -s 1 -n 10

#running on my own mac
#python3 svm.py -d /Users/yuxuanhong/Documents/CS_360_ML/Final_Project/credit_data/creditcard.csv  -n 10
#python3 adaboost.py -d /Users/yuxuanhong/Documents/CS_360_ML/Final_Project/credit_data/creditcard.csv -t 0.5 -n 10
#python3 run_pipeline.py -d /Users/yuxuanhong/Documents/CS_360_ML/Final_Project/credit_data/creditcard.csv -s 1 -n 10
def main():
    opts = util.parse_args()
    X, y = util.data_load(opts.dataset)
    #set upsample range
    n = opts.upsamplen if opts.upsamplen is not None else 1
    start = n if opts.upsamplestart is None else 1

    if start > n:
        print("unsample range error")
        sys.exit()
    for i in np.arange(start,n+1):
        print("i",i)
        needed=util.needed_n(X,y,i)
        print("needed",needed)
        print("pre-upsampled len x",len(X))
        upsampled_X,upsampled_y=util.upsample(X,y,needed)
        print("length of upsampled_X",len(upsampled_X))
        #use a good param to improve speed
        X_train, X_test, y_train, y_test = train_test_split(upsampled_X,upsampled_y,test_size=0.3,random_state=42)
        X_train, X_test = util.normalize(X_train, X_test)
        model=SVC(C=1000,gamma=0.1)
        model.fit(X_train, y_train)
        predictions=model.predict(X_test)
        cm=confusion_matrix(y_test,predictions)
        report=classification_report(y_test,predictions)

        #precision-recall curve
        y_score = model.decision_function(X_test)
        average_precision = average_precision_score(y_test, y_score)
        disp=plot_precision_recall_curve(model,X_test,y_test)
        disp.ax_.set_title('Precision-Recall curve with 10x upsampling: '
                   'AP={0:0.2f}'.format(average_precision))
        plt.show()

        #roc curve
        # svc_roc=plot_roc_curve(model,X_test,y_test)
        # svc_roc.ax_.set_title('SVM ROC Curve with 10x upsampling')
        # plt.show()

        print(cm)
        print(report)

    '''
    #use grid search to create & train models
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
    '''

if __name__ == '__main__':
    main()