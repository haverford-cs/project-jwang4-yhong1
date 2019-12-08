# Final project for CS360

## Development Note:
### 2019/12/04
- Add data parsing, tested on temp.csv (a snapshot of original file)
- Initialization of fully connected network with two dense layers

### 2019/12/05
- All data points are detected as no-fraud, not acceptable
- Normalize data and delete one feature
- NN reaches a true positive of 65%
- Will start working on using SVM, Adaboost

### 2019/12/07
- Initialization of Adaboost, havent used different threshold yet
- Rather than using only 100,000 data, models are trained on the whole dataset this time
- Both Adaboost and nn reach a true positive rate ranging from 70 to 75%
- No idea why we can reach this result, need further investigation

