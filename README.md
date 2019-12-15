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
- Decided to run SVM twice - one with gammaa=1 and one without to see the difference in accuracy & runtime (Concluded that we need gamma=1)
- Investigated the built-in Roc curve functions for our use

### 2019/12/09 (Jiaping)
- Add upsample function to manually increase weight on examples with true label
- Performance gets worse on adaboost with a ratio of 0.3
- Add roc curve

### 2019/12/10 (Andy)
- Get rid of kfolds for SVM & still run single SVM multiple times to finish collecting results without upsampling

### 2019/12/10 (Jiaping)
- Add another type of upsample, this time we only increase the number of true class by n times
- Fix the issue with upsample
- Adaboost reaches a recall rate of 0.85 when upsampling true class by 10 times
- NN reaches a recall rate of 0.798 when upsampling true class by 10 times
- Plan to enable adjusting threshold for Adaboost
- Plan to make roc curve on models with different upsample ratio 

### 2019/12/12 (Andy)
- Clean up the util.py to make some functions more understandable & Added comments to each method to make it more readable for future readers
- Update SVM to accommodate upsamples
- Will run SVM with 10X upsample to see the difference in performance
- will create a results section in README that contains the running results

### 2019/12/13 (Jiaping)
- Revise comment style
- Modify parser and right now we can run adaboost with a threshold range
- Users can now run models with a range of upsample rate
- Start to produce final result and plan to delete intermediate code 

### 2019/12/12 (Andy)
- Fix an error in util parsing - changing the type for '--upsamplen' from 'float' to 'int' to fix the err
- Modified SVM to accomodate new parsing features
- Added plot_recall_upsample_curve funciton
- Get all results from running SVM
- Start to plot all models on the recall_upsample curve




## Model Output Results:
# SVM:


# adaboost:


# fcnn: