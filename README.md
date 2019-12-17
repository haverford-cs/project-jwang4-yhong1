# Final project for CS360
# Data- Use this website to download the dataset
https://www.kaggle.com/mlg-ulb/creditcardfraud#creditcard.csv

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

### 2019/12/14 (Andy)
- change the type for '--upsamplen' from 'float' to 'int'
- Modified SVM to accomodate new parsing features
- Added plot_recall_upsample_curve funciton
- Get all results from running SVM
- Finish ploting all models on the recall_upsample curve
- create generate_curves.py for creating recall_upsample curve
- all results updated. Ready to start working on presentation slides tomorrow

### 2019/12/16 (Andy)
- Add roc curve & precision recall curves to SVM


## Model Output Results:
### SVM:
#running SVM with 1-10 upsample 

#upsample=1, recall=0.78
[[85265    11]
 [   36   131]]


#upsample=2, recall=0.82
[[85268    11]
 [   56   255]]


#upsample=3, recall=0.81
[[85306     7]
 [   81   344]]


#upsample=4, recall=0.84
[[85292    12]
 [   95   486]]


#upsample=5, recall=0.82
[[85290    12]
 [  134   597]]


#upsample=6, recall=0.81
[[85288    13]
 [  166   714]]

#upsample=7, recall=0.82
[[85300    11]
 [  186   831]]


#upsample=8, recall=0.85
[[85271    24]
 [  174  1007]]


#upsample=9, recall=0.84
[[85329    23]
 [  206  1065]]


#upsample=10, recall=0.84  - 1.53%
[[85279    31]
 [  233  1228]]


#===
#upsample=20, recall=0.87  - 3.17%
[[85292    56]
 [  383  2516]]

#positive case - 4.77%
#upsample=30, recall=0.88
[[85293    58]
 [  541  3831]]

#10%
#upsample=60, recall=0.9
[[85224    103]
 [  888  7936]]

### adaboost:
Upsample range from 1 time to times:
[[85293    13] [   36   101]] [[85290    18] [   56   226]] [[85287    33] [   76   342]] [[85310    40] [  106   429]] [[85321    36] [  123   553]] [[85299    49] [  118   715]] [[85294    44] [  145   845]] [[85264    50] [  193   969]] [[85326    41] [  208  1048]] [[85259    63] [  214  1235]]

Unsample param = 10:
0.4
[[19782 65531]
 [    0  1458]]
0.405
[[19982 65331]
 [    0  1458]]
0.410
[[20020 65293]
 [    0  1458]]
0.415
[[20026 65287]
 [    0  1458]]
0.420
[[20026 65287]
 [    0  1458]]
0.425
[[20026 65287]
 [    0  1458]]
0.430
[[20028 65285]
 [    0  1458]]
0.435
[[20033 65280]
 [    0  1458]]
0.440
[[20064 65249]
 [    0  1458]]
0.445
[[20226 65087]
 [    0  1458]]
0.450
[[20796 64517]
 [    0  1458]]
0.455
[[22545 62768]
 [    0  1458]]
0.460
[[26687 58626]
 [    0  1458]]
0.465
[[35198 50115]
 [    0  1458]]
0.470
[[48167 37146]
 [    4  1454]]
0.475
[[63503 21810]
 [    4  1454]]
0.480
[[76080  9233]
 [    4  1454]]
0.485
[[82029  3284]
 [    4  1454]]
0.490
[[84409   904]
 [   30  1428]]
0.495
[[85091   222]
 [  134  1324]]
0.500
[[85247    66]
 [  225  1233]]
0.505
[[85286    27]
 [  317  1141]] 


### fcnn:
Upsample range from 1 time to times:
[array([[85273,    11],
       [   55,   104]]), 
       array([[85269,    23],
       [   63,   235]]), 
       array([[85279,    34],
       [   68,   357]]), 
       array([[85300,    24],
       [  108,   453]]), 
       array([[85300,    29],
       [  123,   581]]), 
       array([[85289,    26],
       [  137,   729]]), 
       array([[85292,    40],
       [  136,   860]]), 
       array([[85253,    36],
       [  223,   964]]), 
       array([[85353,    21],
       [  238,  1011]]), 
       array([[85267,    42],
       [  209,  1253]])]
