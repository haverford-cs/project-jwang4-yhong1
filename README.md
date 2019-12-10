# Final project for CS360
# Lab Notebook

## Development Note:
### 2019/12/04 (Jiaping)
- Add data parsing, tested on temp.csv (a snapshot of original file)
- Initialization of fully connected network with two dense layers

### 2019/12/05 (Jiaping)
- All data points are detected as no-fraud, not acceptable
- Normalize data and delete one feature
- NN reaches a true positive of 65%
- Will start working on using SVM, Adaboost

### 2019/12/07 (Jiaping)
- Initialization of Adaboost, havent used different threshold yet
- Rather than using only 100,000 data, models are trained on the whole dataset this time
- Both Adaboost and nn reach a true positive rate ranging from 70 to 75%
- No idea why we can reach this result, need further investigation

### 2019/12/08 (Andy)
- Note that at this point dataset has not been upsampled yet - (meaning repopulate fraud cases to reach 4k cases). In other words, we still use the original data here
- Create SVM
- Figuring out the params to be used for SVM 

### 2019/12/09 (Andy)
- Figure out & clean up correct params for SVM
- Decided to run SVM twice - one with gammaa=1 and one without to see the difference in accuracy & runtime 
- Implment Roc curve functions that can be used across all model methods

### 2019/12/09 (Jiaping)
- Add upsample function to manually increase weight on examples with true label
- Performance gets worse on adaboost with a ratio of 0.3
- Add roc curve

