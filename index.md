# Practical Machine Learning Course Project

## Introduction

In this project we use  from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways.The goal is to predict the manner in which they did the exercise. This is the "classe" variable in the training set. 

[The data for this project come from Groupware@LES website :] (http://groupware.les.inf.puc-rio.br/har).

## Data Collection


```r
## group url : http://groupware.les.inf.puc-rio.br/har 
furl1 <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
download.file(furl1,"train.csv")
furl2 <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
download.file(furl2,"test.csv")
```


## Cleaning Test and Train Data

By inspection, There exist many data columns containing a high proportion of missing values. In addtion, the first seven columns are irrelevent to the prediction task (those are about user names, timestamps windows and row indexes  and so furths). Consequently, we remove such colums from the data used the analysis, since such columns have no added value to the prediction model we want.

Both traing and testing data are prepared in the same way.


```r
set.seed(2222)
## read from csv files
data_training    <- read.csv("train.csv",na.strings = c("","NA"," "))
data_testing    <- read.csv("test.csv",na.strings = c("","NA"," "))

## removing NA columns
na <-apply(data_training, 2, function(v) sum(is.na(v)))
keep_col <- which(na == 0)
tidy_train_data <- data_training[,keep_col]
tidy_test_data <- data_testing[,keep_col]

## removing irrelevant columns (7 ) : X , user_name, ..timestamp .. , new_window ..
tidy_train_data <- tidy_train_data[,-1*c(1:7)]
tidy_test_data <- tidy_test_data[,-c(1:7)]
```

## Data Slicing  

Since the size of the original data in fairly large, we make the size of the training set as _60%_ of that of the original data and the size of the validation data is _40%_  


```r
library(caret)
```

```
## Loading required package: lattice
```

```
## Loading required package: ggplot2
```

```r
inTrain<-createDataPartition(y=tidy_train_data$classe,p=0.6,list=FALSE)
## training set
cv_train<-tidy_train_data [inTrain,]
##validation set
out_test <-tidy_train_data [-inTrain,]
```

## Model Building  

### Model selection  

Now we are left with 52 predictors representin the various kinds measurements coming from body sensors, and the outcome class of the excercise. _Random forest_ model is choose as the method of classification for several reasons :

 1- it is well suited to handle a large number of features (in this case 52)
 2- it can be used to estimate variable importance (so we can choose the best ones)
 3- it is more robust to the effect of correlated predictors


### Training Options  

The resampling method used is *5 fold* cross validation rather than the default  bootstrapping, in order to improve processing performance and in the same time the accuracy is preserved.  


```r
library(doMC)
registerDoMC(cores = 2)

start <- date()
rf_model<-train(classe~.,data=cv_train,method="rf",
                trControl=trainControl(method="cv",number=5))
end <-date()

print(rf_model)
save(rf_model, file = "rf_model.RData")
rm(rf_model)
```


```
## Random Forest 
## 
## 11776 samples
##    52 predictors
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (5 fold) 
## Summary of sample sizes: 9421, 9421, 9420, 9422, 9420 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa      Accuracy SD  Kappa SD   
##    2    0.9853942  0.9815188  0.001814612  0.002295298
##   27    0.9869226  0.9834541  0.001599717  0.002025773
##   52    0.9763083  0.9700248  0.002926735  0.003702544
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 27.
```

## Measuring performance

The accuracy of the model can calculated from the hold out test data. In addition we can use the cross validation to estimate accuracy from the training data. In this section both method used then we compare them.  

### Accuracy based on the validation set (from hold out set)

First we measure the accuracy (hence the out of sample error) based on the test(unseen set).
This accuracy reprensents the measured accuracy and next we calculate the an estimation to the accuracy by cross validation.  

### Confusion Matrix of Test data prediction


```r
testPred <- predict(rf_model, out_test[,-53])
conf <- confusionMatrix(testPred , out_test$classe)
conf$table
```

```
##           Reference
## Prediction    A    B    C    D    E
##          A 2232    8    0    0    0
##          B    0 1508    3    1    0
##          C    0    2 1360    4    0
##          D    0    0    5 1281    0
##          E    0    0    0    0 1442
```

```r
print(conf$overall,digits = 3)
```

```
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##          0.997          0.996          0.996          0.998          0.284 
## AccuracyPValue  McnemarPValue 
##          0.000            NaN
```

### ** Out of Sample Error ** 


```r
accuracy.measured<- round( postResample(testPred, out_test$classe)[1],3)
## out of sampl error measured 
oose.measured <- round((1- accuracy.measured)* 100,3)
names(oose.measured) <- "OutOfSampleError.measured %"
```

### Accuracy estimation based on Cross validaation 


```r
#accuracy.estimation <- mean(rf_model$resample$Accuracy)
accuracy.estimation<-round(getTrainPerf(rf_model)["TrainAccuracy"],3)
print(accuracy.estimation,digits = 3)
```

```
##   TrainAccuracy
## 1         0.987
```

```r
## out of sampl error estimated from cross validation
oose.estimated <- round((1 - accuracy.estimation) * 100,3)

names(oose.estimated) = "OutOfSampleError.estimated %"
print(oose.estimated,digits = 3)
```

```
##   OutOfSampleError.estimated %
## 1                          1.3
```

The error results shows that cross validation error is a good approximation of out of sample error.


Metric         |  Test set               | Cross validation
-------------        | -------------           | -------------
Accuracy             | 0.997   | 0.987
Out of Sample error  | 0.3%     | 1.3%


## Model results 

The random forest model can provide a quantitative evaluation to the importance of each predictor in building the classifier, hence we can select/understand  the most important/discriminant predictors.

Here a plot of the top most important features 

```r
predictors.Imp <- varImp(rf_model, scale = FALSE)
```

```
## Loading required package: randomForest
```

```
## randomForest 4.6-12
```

```
## Type rfNews() to see new features/changes/bug fixes.
```

```
## 
## Attaching package: 'randomForest'
```

```
## The following object is masked from 'package:ggplot2':
## 
##     margin
```

```r
plot(predictors.Imp, top = 20)
```

![](index_files/figure-html/unnamed-chunk-8-1.png)\

### Predicting the Unknown 20 Test Case  

* The accuracy of prediction from the measured   accuracy is : __accuracy.measured^20__ = 0.9416796  
* The accuracy of prediction from the estimated  accuracy is : __accuracy.estimation^20__= 0.7697382 

But actually the all the 20 test cases has been correctly classified  


```r
predicted.cases <- predict(rf_model,tidy_test_data[,-length(tidy_test_data)])
df <- cbind(index = 1:20, class = as.character(predicted.cases))
library(knitr)
kable(df, format = "markdown")
```



|index |class |
|:-----|:-----|
|1     |B     |
|2     |A     |
|3     |B     |
|4     |A     |
|5     |A     |
|6     |E     |
|7     |D     |
|8     |B     |
|9     |A     |
|10    |A     |
|11    |B     |
|12    |C     |
|13    |B     |
|14    |A     |
|15    |E     |
|16    |E     |
|17    |A     |
|18    |B     |
|19    |B     |
|20    |B     |

