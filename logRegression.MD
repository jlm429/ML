---
title: 'Logistic Regression'
subtitle: |-
  Data and Visual Analytics - Calculus based logistic regression
output:
  RMD (R Markdown) 
---

#TODO: Fix MD lc graph rendering

#1. Data Preprocessing [20 points]

```{r}
#import
library(readr)
mnist_train <- read_csv("~/mnist/mnist/mnist_train.csv", col_names = FALSE)

mnist_test <- read_csv("~/mnist/mnist/mnist_test.csv", col_names = FALSE)
```

```{r}
#partition

test_0_1=mnist_test[,!(unlist(mnist_test[785,]))==3]
test_0_1=test_0_1[,!(unlist(test_0_1[785,]))==5]
train_0_1=mnist_train[,!(unlist(mnist_train[785,]))==3]
train_0_1=train_0_1[,!(unlist(train_0_1[785,]))==5]

test_3_5=mnist_test[,!(unlist(mnist_test[785,]))==0]
test_3_5=test_3_5[,!(unlist(test_3_5[785,]))==1]
train_3_5=mnist_train[,!(unlist(mnist_train[785,]))==0]
train_3_5=train_3_5[,!(unlist(train_3_5[785,]))==1]

#print dim
print(dim(mnist_test))
print(dim(mnist_train))
print(dim(test_0_1))
print(dim(train_0_1))
print(dim(test_3_5))
print(dim(train_3_5))
```

```{r}

#sep class labels
test_label_0_1=test_0_1[785, ]
test_0_1=test_0_1[-785,]
train_label_0_1=train_0_1[785, ]
train_0_1=train_0_1[-785,]

test_label_3_5=test_3_5[785, ]
test_3_5=test_3_5[-785,]
train_label_3_5=train_3_5[785, ]
train_3_5=train_3_5[-785,]

#function for visualizing data
visualizeData = function(data){
  df=data.frame(matrix(, nrow=28, ncol=28))
  for (num in seq(1, 28, 1)){
     for (num2 in seq(1, 28, 1))
         df[num2, num]=data[num2+28*num-28,1]
  }
  rotate <- function(x) t(apply(x, 2, rev))
  df=as.matrix(df)
  return(image(rotate(df), col=gray(0:255/255)))
}

#visualize 1 image from each class (0,1,3,5)
#blah=matrix(test_0_1[,1], 255, byrow=TRUE)
#convert to 255x255 matrix (use loop if unable to find vectorized solution)

#image 0
image0=visualizeData(test_0_1[,500])
image0Label=test_label_0_1[,500]
image0Title=paste("Actual Label=", as.numeric(image0Label))
title(main = image0Title, font.main = 4)

#image 1
image1=visualizeData(test_0_1[,1625])
image1Label=test_label_0_1[,1625]
image1Title=paste("Actual Label=", as.numeric(image1Label))
title(main = image1Title, font.main = 4)

#image 3
image3=visualizeData(test_3_5[,800])
image3Label=test_label_3_5[,800]
image3Title=paste("Actual Label=", as.numeric(image3Label))
title(main = image3Title, font.main = 4)

#image 5
image5=visualizeData(test_3_5[,1700])
image5Label=test_label_3_5[,1700]
image5Title=paste("Actual Label=", as.numeric(image5Label))
title(main = image5Title, font.main = 4)



```


2. Theory 


# Formula for the loss function used in Logistic Regression (expression to be minimized).

From the tutorial - "Fun with Logistic Regression"
http://nbviewer.jupyter.org/gist/vietjtnguyen/6655020

Q. Formula for the loss function used in Logistic Regression (expression to be minimized):

<!-- $L(\theta) =  \underset{\theta}{\operatorname{argmin}}  \frac{1}{N}  \sum_{n=1}^{N} ln (1 + e ^{ y_{n}  \theta^{T} x_{n}  })$ -->
![Loss Function](https://github.com/jlm429/ML/blob/master/images/lossFunction2.png)

Where $x_{n}$ = sample n from input matrix x and $\theta^{T}_{i} x_{n} = \sum_{i=1}^{d} x_{i}\theta_{i}$ (d=number of attributes + 1 for offset)

Q. Derivation of the gradient of the loss function with respect to model parameters: 

<!-- $-(\frac{1}{N}  \sum_{n=1}^{N} \frac{y_{n}x_{n}}{1 + e ^{ y_{n}  \theta^{T}_{i} x_{n}  }})$ -->
![Gradient](https://github.com/jlm429/ML/blob/master/images/gradient.png)

Where $x_{n}$ = sample n from input matrix x and $\theta^{T}_{i} x_{n} = \sum_{i=1}^{d} x_{i}\theta_{i}$ (d=number of attributes + offset)


Q. Expression of the Stochastic Gradient Descent (SGD) update rule based on loss function that using a single sample x(i), y(i) at a time: 
![SGD](https://github.com/jlm429/ML/blob/master/images/SGD.PNG)


<!-- $\theta_{i+1}=\theta_{i}+\alpha (\frac{1}{N}  \sum_{n=1}^{N} \frac{y_{n}x_{n}}{1 + e ^{ y_{n}  \theta^{T}_{i} x_{n}  }})$ -->

Where $x_{n}$ = sample n from input matrix x and $\theta^{T}_{i} x_{n} = \sum_{i=1}^{d} x_{i}\theta_{i}$ (d=number of attributes + offset)


Q. Pseudocode for training a model using Logistic Regression and SGD.

A. 
for i in iterations{
 subsetIndex=randomSample(nrow(data))
 for n in number of samples{
	gradDescent=gradDescent + label[subsetIndex, n]*data[subsetIndex, n]/(1+e^(labelMat[subsetIndex, n]*sum(theta[subsetIndex]*data[subsetIndex,n])
  }
  theta=theta+alpha*(gradientDescent/N)
}

Q. Estimate of the number of operations per epoch of SGD, where an epoch is one complete iteration through all the training samples (Big-O notation, in terms of the number of samples n, and the dimensionality of each sample d).  


Assuming mult/add/div are constant time and operations are vectorized by sample - the overall runtime of SGD would be 
Iterations*n samples = O(n) for each iteration.  

#3 Implementation 

```{r}
#SGD
SGD = function(data, labels, theta, alpha){
  
  #number of samples
  N=ncol(data)
  
  #create label matrix
  labelMat=matrix(labels)
  for (num in seq(1, nrow(data)-1, b=1)){
    labelMat=cbind(labelMat, labelMat[,1])
  }
  
  labelMat=t(labelMat)
  print('labelmatrix dim/data dim')
  print(dim(labelMat))
  print(dim(data))
  print(dim(theta))
  cat('dim labels=', dim(labels))
  
  #vectorized version of gradient descent 
  
  iters=1
  for (num2 in seq(1,iters, 1)){
    #Gradient Descent
    subsetSGD=as.vector(seq(1, nrow(data), 1))
    #uncomment for stachastic gradient descent
    subsetSGD=as.vector(round(runif(nrow(data)/5, 1, nrow(data)-1)))
    gradientDescent=0
    for (num in seq(1, N, 1)){
      gradientDescent=gradientDescent+ ((as.numeric(labelMat[subsetSGD,num])*data[subsetSGD,num])/(1+exp(as.numeric(labelMat[subsetSGD,num])*sum(theta[subsetSGD]*data[subsetSGD,num]))))
    }
  theta=theta+alpha*((as.numeric(as.matrix(gradientDescent))/N))
  }
  return(theta)
}

#sigmoid
sigmoid = function(num, cutoff){
  
  #output= 1.0/(1.0+exp(-num))
  if (num>cutoff){output=0}
  else{output=1}
  return(output)
}

#convert labels to 0/1
convertLabels = function(labels){
  output= labels
  for (num in seq(1, ncol(labels),1)){
    if (as.numeric(labels[1,num]==3)){output[1,num]=0}
    if (as.numeric(labels[1,num]==5)){output[1,num]=1}
  }
  return(output)
}

#convert labels
convertLabelsBack = function(labels){
  output= labels
  for (num in seq(1, ncol(labels),1)){
    if (as.numeric(labels[1,num]==0)){output[1,num]=3}
    if (as.numeric(labels[1,num]==1)){output[1,num]=5}
  }
  return(output)
}

#create n random values from 0-1
randomValues = function(num){
  
  rand=runif(num, 0, 1)
  return(rand)
}

#train 
train = function(data, labels, alpha){

  # append offset column to data
  train_data=data
  offset=matrix(1, 1, ncol(data))
  colnames(offset)=colnames(train_data)
  train_data=rbind(train_data, offset)
  
  #convert labels
  train_labels=labels
  train_labels=convertLabels(train_labels)

  #theta = offset + random values
  #theta=matrix(randomValues(nrow(train_data)-1))
  theta=matrix(0, nrow(train_data)-1, 1)
  theta=rbind(theta,1)

  #run SGD
  theta=SGD(train_data, train_labels, theta, alpha)
  

  return(theta)

}

predict = function(theta, data){
  
  #init y_predict and multiply data * theta for prediction values
  train_data=data
  
  
  y_predict = matrix(0, 1, ncol(train_data))
  for (num in seq(1, ncol(train_data), b=1)){
    y_predict[1,num] = sum(train_data[,num]*theta)
  }
  
  cutoff=median(as.numeric(y_predict))
  #run through sigmoid
    for (num in seq(1, ncol(y_predict), b=1)){
    y_predict[1,num] = sigmoid(y_predict[1,num], cutoff)
  }

  #labels=convertLabelsBack(y_predict)
  labels=y_predict
  return(labels)
}


```

#3 Continued

Implementation notes, e.g. initialization method, convergence criteria, modifications made to the input data, algorithm, etc. Visualization of 2 correct and 2 incorrect predictions each for 0/1 and 3/5 training sets (a total of 8 samples).

Theta initialized to all 0's and offset term appended to both data and theta.  Convergence is based on iterations of SGD and labels are mapped to values of 0 or 1.  Training is based on a random subset of samples from each training set.  

#Visualize 2 correct/2 incorrect predictions from 0/1 & 3/5
```{r}

#return 
getIndexOfAccuratePredictions = function(labels, labels_pred){
  labels=convertLabels(labels)
  labels_pred=convertLabels(labels_pred)
  totalLabels=ncol(labels)
  accMat=matrix(labels)
  accMat=cbind(accMat, matrix(labels_pred))
  accIndexMatrix=matrix(0,1,1)
  for(num in seq(1, nrow(accMat), 1)){
    if(as.numeric(accMat[num, 1])==as.numeric(accMat[num,2])){
      accIndexMatrix=rbind(accIndexMatrix, num)
    }
    #else{labelsPlusPredict[num]=0}
    
  }
  accIndexMatrix=accIndexMatrix[-1,]
  return(accIndexMatrix)
}

getIndexOfInaccuratePredictions = function(labels, labels_pred){
  labels=convertLabels(labels)
  labels_pred=convertLabels(labels_pred)
  totalLabels=ncol(labels)
  accMat=matrix(labels)
  accMat=cbind(accMat, matrix(labels_pred))
  accIndexMatrix=matrix(0,1,1)
  for(num in seq(1, nrow(accMat), 1)){
    if(as.numeric(accMat[num, 1])!=as.numeric(accMat[num,2])){
      accIndexMatrix=rbind(accIndexMatrix, num)
    }
    #else{labelsPlusPredict[num]=0}
    
  }
  accIndexMatrix=accIndexMatrix[-1,]
  return(accIndexMatrix)
}


#0/1

#train theta 
alpha=.005
#sample_size=round(.0005* ncol(train_0_1))
sample_size=500
randomTrainInd0_1=sample(ncol(train_0_1), sample_size)
randomTestInd0_1=sample(ncol(test_0_1), sample_size)
train_data_0_1=train_0_1[,randomTrainInd0_1]
test_data_0_1=test_0_1[,randomTestInd0_1]
test_labels_0_1=test_label_0_1[, randomTestInd0_1]
train_labels_0_1=train_label_0_1[, randomTrainInd0_1]
theta=train(train_data_0_1, train_labels_0_1, alpha)

#predict on test data
predict01=predict(theta, test_data_0_1)
#accuracy01=accuracy(test_labels_0_1,predict01)
indexCorrect=matrix(getIndexOfAccuratePredictions(test_labels_0_1,predict01))
indexIncorrect=matrix(getIndexOfInaccuratePredictions(test_labels_0_1,predict01))

labelsPredict=cbind(matrix(test_labels_0_1), matrix(predict01))

#Visualize 2 - 0/1 correct
correct1=visualizeData(test_data_0_1[,indexCorrect[1,1]])
correct1Label=test_labels_0_1[,indexCorrect[1,1]]
correct1Predict=predict01[,indexCorrect[1,1]]
correct1Title=paste("Actual Label=", as.numeric(correct1Label), " Predicted Label=", as.numeric(correct1Predict))
title(main = correct1Title, font.main = 4)

correct2=visualizeData(test_data_0_1[,indexCorrect[2,1]])
correct2Label=test_labels_0_1[,indexCorrect[2,1]]
correct2Predict=predict01[,indexCorrect[2,1]]
correct2Title=paste("Actual Label=", as.numeric(correct2Label), " Predicted Label=", as.numeric(correct2Predict))
title(main = correct2Title, font.main = 4)

#Visualize 2 - 0/1 incorrect
incorrect1=visualizeData(test_data_0_1[,indexIncorrect[1,1]])
incorrect1Label=test_labels_0_1[,indexIncorrect[1,1]]
incorrect1Predict=predict01[,indexIncorrect[1,1]]
incorrect1Title=paste("Actual Label=", as.numeric(incorrect1Label), " Predicted Label=", as.numeric(incorrect1Predict))
title(main = incorrect1Title, font.main = 4)

incorrect2=visualizeData(test_data_0_1[,indexIncorrect[2,1]])
incorrect2Label=test_labels_0_1[,indexIncorrect[2,1]]
incorrect2Predict=predict01[,indexIncorrect[2,1]]
incorrect2Title=paste("Actual Label=", as.numeric(incorrect2Label), " Predicted Label=", as.numeric(incorrect2Predict))
title(main = incorrect2Title, font.main = 4)


#3/5

alpha=.005
#sample_size=round(.0005* ncol(train_3_5))
sample_size=500
randomTrainInd3_5=sample(ncol(train_3_5), sample_size)
randomTestInd3_5=sample(ncol(test_3_5), sample_size)
train_data_3_5=train_3_5[,randomTrainInd3_5]
test_data_3_5=test_3_5[,randomTestInd3_5]
test_labels_3_5=test_label_3_5[, randomTestInd3_5]
train_labels_3_5=train_label_3_5[, randomTrainInd3_5]
theta=train(train_data_3_5, train_labels_3_5, alpha)

#predict on test data
predict35=predict(theta, test_data_3_5)
#accuracy35=accuracy(test_labels_3_5,predict35)
indexCorrect=matrix(getIndexOfAccuratePredictions(test_labels_3_5,predict35))
indexIncorrect=matrix(getIndexOfInaccuratePredictions(test_labels_3_5,predict35))

labelsPredict=cbind(matrix(test_labels_3_5), matrix(predict35))

#Visualize 2 - 3/5 correct
correct1_35=visualizeData(test_data_3_5[,indexCorrect[1,1]])
correct1Label_35=test_labels_3_5[,indexCorrect[1,1]]
correct1Predict_35=convertLabelsBack(matrix(predict35[,indexCorrect[1,1]]))
correct1Title_35=paste("Actual Label=", as.numeric(correct1Label_35), " Predicted Label=", as.numeric(correct1Predict_35))
title(main = correct1Title_35, font.main = 4)

correct2_35=visualizeData(test_data_3_5[,indexCorrect[4,1]])
correct2Label_35=test_labels_3_5[,indexCorrect[4,1]]
correct2Predict_35=convertLabelsBack(matrix(predict35[,indexCorrect[4,1]]))
correct2Title_35=paste("Actual Label=", as.numeric(correct2Label_35), " Predicted Label=", as.numeric(correct2Predict_35))
title(main = correct2Title_35, font.main = 4)

#Visualize 2 - 3/5 incorrect
incorrect1_35=visualizeData(test_data_3_5[,indexIncorrect[1,1]])
incorrect1Label_35=test_labels_3_5[,indexIncorrect[1,1]]
incorrect1Predict_35=convertLabelsBack(matrix(predict35[,indexIncorrect[1,1]]))
incorrect1Title_35=paste("Actual Label=", as.numeric(incorrect1Label_35), " Predicted Label=", as.numeric(incorrect1Predict_35))
title(main = incorrect1Title_35, font.main = 4)

incorrect2_35=visualizeData(test_data_3_5[,indexIncorrect[2,1]])
incorrect2Label_35=test_labels_3_5[,indexIncorrect[2,1]]
incorrect2Predict_35=convertLabelsBack(matrix(predict35[,indexIncorrect[2,1]]))
incorrect2Title_35=paste("Actual Label=", as.numeric(incorrect2Label_35), " Predicted Label=", as.numeric(incorrect2Predict_35))
title(main = incorrect2Title_35, font.main = 4)

```



#4. Modeling

```{r}

accuracy = function(labels, labels_pred){
  labels=convertLabels(labels)
  labels_pred=convertLabels(labels_pred)
  totalLabels=ncol(labels)
  accMat=matrix(labels)
  accMat=cbind(accMat, matrix(labels_pred))
  labelsPlusPredict=0
  for(num in seq(1, nrow(accMat), 1)){
    if(as.numeric(accMat[num, 1])==as.numeric(accMat[num,2])){
      labelsPlusPredict[num]=1
    }
    else{labelsPlusPredict[num]=0}
    
  }
  #labelPlusPredict=as.numeric(accMat[,1])+as.numeric(accMat[,2])
  #accMat=cbind(accMat, labelPlusPredict)
  #accMat=accMat[!(accMat[,3]==1),]
  #percentCorrectLabels=nrow(accMat)/totalLabels
  acc=sum(labelsPlusPredict)/nrow(accMat)
  return(acc)
}

model = function(train_data, train_labels, test_data, test_labels, alpha){
 
  #training data
  theta=train(train_data, train_labels, alpha)
  train_predict=predict(theta, train_data)
  train_acc=accuracy(train_labels, train_predict)
  
  #test data
  test_predict=predict(theta, test_data)
  test_acc=accuracy(test_labels, test_predict)
  
  return(c(theta, train_acc, test_acc))
}

#3_5 modeling test


alpha=.01
#sample_size=round(.0005* ncol(train_3_5))
sample_size=100
randomTrainInd3_5=sample(ncol(train_3_5), sample_size)
randomTestInd3_5=sample(ncol(test_3_5), sample_size)
train_data_3_5=train_3_5[,randomTrainInd3_5]
test_data_3_5=test_3_5[,randomTestInd3_5]
test_labels_3_5=test_label_3_5[, randomTestInd3_5]
train_labels_3_5=train_label_3_5[, randomTrainInd3_5]

results_3_5=model(train_data_3_5, train_labels_3_5, test_data_3_5, test_labels_3_5, alpha)

#0_1 modeling test

alpha=.001
#sample_size=round(.0005* ncol(train_0_1))
sample_size=100
randomTrainInd0_1=sample(ncol(train_0_1), sample_size)
randomTestInd0_1=sample(ncol(test_0_1), sample_size)
train_data_0_1=train_0_1[,randomTrainInd0_1]
test_data_0_1=test_0_1[,randomTestInd0_1]
test_labels_0_1=test_label_0_1[, randomTestInd0_1]
train_labels_0_1=train_label_0_1[, randomTrainInd0_1]
results_0_1=model(train_data_0_1, train_labels_0_1, test_data_0_1, test_labels_0_1, alpha)


#model both 10 times with different alpha values


model_0_1 = function(){
results_0_1_Mat=matrix(c(0,0))
results_0_1_Mat=(t((results_0_1_Mat)))
#0_1 
for (num in seq(1,10,1)){
  alpha=num*.01
  sample_size=round(.05* ncol(train_0_1))
  randomTrainInd0_1=sample(ncol(train_0_1), sample_size)
  randomTestInd0_1=sample(ncol(test_0_1), sample_size)
  train_data_0_1=train_0_1[,randomTrainInd0_1]
  test_data_0_1=test_0_1[,randomTestInd0_1]
  test_labels_0_1=test_label_0_1[, randomTestInd0_1]
  train_labels_0_1=train_label_0_1[, randomTrainInd0_1]
  results_0_1=model(train_data_0_1, train_labels_0_1, test_data_0_1, test_labels_0_1, alpha)
  results_0_1=matrix(results_0_1)
  results_0_1_Mat=rbind(results_0_1_Mat, c(results_0_1[786,], results_0_1[787,]))
  
}

results_0_1_Mat=results_0_1_Mat[-1,]
colnames(results_0_1_Mat)=c('train_acc', 'test_acc')
return(results_0_1_Mat)

}


model_3_5 = function(){
results_3_5_Mat=matrix(c(0,0))
results_3_5_Mat=(t((results_3_5_Mat)))
#3_5 
for (num in seq(1,10,1)){
  alpha=num*.01
  sample_size=round(.05* ncol(train_3_5))
  randomTrainInd3_5=sample(ncol(train_3_5), sample_size)
  randomTestInd3_5=sample(ncol(test_3_5), sample_size)
  train_data_3_5=train_3_5[,randomTrainInd3_5]
  test_data_3_5=test_3_5[,randomTestInd3_5]
  test_labels_3_5=test_label_3_5[, randomTestInd3_5]
  train_labels_3_5=train_label_3_5[, randomTrainInd3_5]
  results_3_5=model(train_data_3_5, train_labels_3_5, test_data_3_5, test_labels_3_5, alpha)
  results_3_5=matrix(results_3_5)
  results_3_5_Mat=rbind(results_3_5_Mat, c(results_3_5[786,], results_3_5[787,]))
  
}

results_3_5_Mat=results_3_5_Mat[-1,]
colnames(results_3_5_Mat)=c('train_acc', 'test_acc')
return(results_3_5_Mat)
}



get01alphaMeans=function(){

  #run model_0_1 10 times 
  results_0_1_DF=model_0_1()
  for(num in seq(1,9,1)){
    newMat=model_0_1()
    results_0_1_DF=cbind(results_0_1_DF, newMat)
  }

  #get 0_1 mean for 10 alpha values
  results_0_1_DF=data.frame(results_0_1_DF)
  model_0_1_trainMean=results_0_1_DF[,seq(1,20, 2)]
  model_0_1_testMean=results_0_1_DF[,seq(2,20, 2)]
  train_01_mean=0
  for (num in seq(1, nrow(model_0_1_trainMean),1)){
    train_01_mean[num]=mean(as.numeric(model_0_1_trainMean[num,]))
  }
  test_01_mean=0
  for (num in seq(1, nrow(model_0_1_testMean),1)){
    test_01_mean[num]=mean(as.numeric(model_0_1_testMean[num,]))
  }
  mean_0_1_df=cbind(train_01_mean, test_01_mean)
  mean_0_1_df=data.frame(mean_0_1_df)
  names(mean_0_1_df)=c("mean_train_acc", "mean_test_acc")
  return(mean_0_1_df)
}

get35alphaMeans=function(){
  #run model_3_5 10 times
  results_3_5_DF=model_3_5()
  for(num in seq(1,9,1)){
    newMat=model_3_5()
    results_3_5_DF=cbind(results_3_5_DF, newMat)
  }

  #get 3_5 mean for 10 alpha values
  results_3_5_DF=data.frame(results_3_5_DF)
  model_3_5_trainMean=results_3_5_DF[,seq(1,20, 2)]
  model_3_5_testMean=results_3_5_DF[,seq(2,20, 2)]
  train_01_mean=0
  for (num in seq(1, nrow(model_3_5_trainMean),1)){
    train_01_mean[num]=mean(as.numeric(model_3_5_trainMean[num,]))
  }
  test_01_mean=0
  for (num in seq(1, nrow(model_3_5_testMean),1)){
    test_01_mean[num]=mean(as.numeric(model_3_5_testMean[num,]))
  }
  mean_3_5_df=cbind(train_01_mean, test_01_mean)
  mean_3_5_df=data.frame(mean_3_5_df)
  names(mean_3_5_df)=c("mean_train_acc", "mean_test_acc")
  return(mean_3_5_df)
}
```

#plot mean for 01/35 alpha value range

```{r}

#mean01DF=get01alphaMeans()
#mean35DF=get35alphaMeans()

#plot(mean01DF$mean_train_acc, ylim=c(.8,.92), main="0/1 Average of 10 - Learning Rate Accuracy", xaxt="n", type="l", lty=4, xlab="alpha", ylab="accuracy", col="blue")
#axis(1, at=1:10, labels=seq(.01, .1, .01))
#lines(mean01DF$mean_test_acc, lty=1, type="l", col="green")
#legend( x="topright", 
#        c("Train", "Test"),
#        col=c("blue", "green"), lty=c(1,1), lwd=4,cex=.5)

#plot(mean35DF$mean_train_acc, ylim=c(.5,.65), main="3/5 Average of 10 - Learning Rate Accuracy", xaxt="n", type="l", lty=4, xlab="alpha", ylab="accuracy", col="blue")
#axis(1, at=1:10, labels=seq(.01, .1, .01))
#lines(mean35DF$mean_test_acc, lty=1, type="l", col="green")
#legend( x="topright", 
#        c("Train", "Test"),
#        col=c("blue", "green"), lty=c(1,1), lwd=4)

```

#Learning Curves

```{r}

#model with 10 different sample sizes


model_0_1_LC = function(){
results_0_1_Mat=matrix(c(0,0))
results_0_1_Mat=(t((results_0_1_Mat)))
#0_1 
for (num in seq(1,10,1)){
  alpha=.01
  sample_size=round(num*.01* ncol(train_0_1))
  randomTrainInd0_1=sample(ncol(train_0_1), sample_size)
  randomTestInd0_1=sample(ncol(test_0_1), sample_size)
  train_data_0_1=train_0_1[,randomTrainInd0_1]
  test_data_0_1=test_0_1[,randomTestInd0_1]
  test_labels_0_1=test_label_0_1[, randomTestInd0_1]
  train_labels_0_1=train_label_0_1[, randomTrainInd0_1]
  results_0_1=model(train_data_0_1, train_labels_0_1, test_data_0_1, test_labels_0_1, alpha)
  results_0_1=matrix(results_0_1)
  results_0_1_Mat=rbind(results_0_1_Mat, c(results_0_1[786,], results_0_1[787,]))
  
}

results_0_1_Mat=results_0_1_Mat[-1,]
colnames(results_0_1_Mat)=c('train_acc', 'test_acc')
return(results_0_1_Mat)

}

get01LCMeans=function(){

  #run model_0_1_LC 5 times 
  results_0_1_DF=model_0_1_LC()
  for(num in seq(1,4,1)){
    newMat=model_0_1_LC()
    results_0_1_DF=cbind(results_0_1_DF, newMat)
  }

  #get 0_1 mean for 5 sample sizes
  results_0_1_DF=data.frame(results_0_1_DF)
  model_0_1_trainMean=results_0_1_DF[,seq(1,10, 2)]
  model_0_1_testMean=results_0_1_DF[,seq(2,10, 2)]
  train_01_mean=0
  for (num in seq(1, nrow(model_0_1_trainMean),1)){
    train_01_mean[num]=mean(as.numeric(model_0_1_trainMean[num,]))
  }
  test_01_mean=0
  for (num in seq(1, nrow(model_0_1_testMean),1)){
    test_01_mean[num]=mean(as.numeric(model_0_1_testMean[num,]))
  }
  mean_0_1_df=cbind(train_01_mean, test_01_mean)
  mean_0_1_df=data.frame(mean_0_1_df)
  names(mean_0_1_df)=c("mean_train_acc", "mean_test_acc")
  return(mean_0_1_df)
}

model_3_5_LC = function(){
results_3_5_Mat=matrix(c(0,0))
results_3_5_Mat=(t((results_3_5_Mat)))
#3_5 
for (num in seq(1,10,1)){
  alpha=.07
  sample_size=round(num*.01* ncol(train_3_5))
  randomTrainInd3_5=sample(ncol(train_3_5), sample_size)
  randomTestInd3_5=sample(ncol(test_3_5), sample_size)
  train_data_3_5=train_3_5[,randomTrainInd3_5]
  test_data_3_5=test_3_5[,randomTestInd3_5]
  test_labels_3_5=test_label_3_5[, randomTestInd3_5]
  train_labels_3_5=train_label_3_5[, randomTrainInd3_5]
  results_3_5=model(train_data_3_5, train_labels_3_5, test_data_3_5, test_labels_3_5, alpha)
  results_3_5=matrix(results_3_5)
  results_3_5_Mat=rbind(results_3_5_Mat, c(results_3_5[786,], results_3_5[787,]))
  
}

results_3_5_Mat=results_3_5_Mat[-1,]
colnames(results_3_5_Mat)=c('train_acc', 'test_acc')
return(results_3_5_Mat)

}

get35LCMeans=function(){

  #run model_3_5_LC 5 times 
  results_3_5_DF=model_3_5_LC()
  for(num in seq(1,4,1)){
    newMat=model_3_5_LC()
    results_3_5_DF=cbind(results_3_5_DF, newMat)
  }

  #get 3_5 mean for 5 sample sizes
  results_3_5_DF=data.frame(results_3_5_DF)
  model_3_5_trainMean=results_3_5_DF[,seq(1,10, 2)]
  model_3_5_testMean=results_3_5_DF[,seq(2,10, 2)]
  train_35_mean=0
  for (num in seq(1, nrow(model_3_5_trainMean),1)){
    train_35_mean[num]=mean(as.numeric(model_3_5_trainMean[num,]))
  }
  test_35_mean=0
  for (num in seq(1, nrow(model_3_5_testMean),1)){
    test_35_mean[num]=mean(as.numeric(model_3_5_testMean[num,]))
  }
  mean_3_5_df=cbind(train_35_mean, test_35_mean)
  mean_3_5_df=data.frame(mean_3_5_df)
  names(mean_3_5_df)=c("mean_train_acc", "mean_test_acc")
  return(mean_3_5_df)
}


#results0_1_LCMAT=get01LCMeans()
#results3_5_LCMAT=get35LCMeans()

#plot(results0_1_LCMAT$mean_train_acc , ylim=c(.8,.92), main="0/1 Average of 5 - Sample Sizes Learning Curve", xaxt="n", type="l", lty=4, xlab="Sample Size (Percent of Training Data)", ylab="accuracy", col="blue")
#axis(1, at=1:10, labels=seq(.01, .1, .01))
#lines(results0_1_LCMAT$mean_test_acc, lty=1, type="l", col="green")
#legend( x="topright", 
#        c("Train", "Test"),
#        col=c("blue", "green"), lty=c(1,1), lwd=4,cex=.5)

#plot(results3_5_LCMAT$mean_train_acc , ylim=c(.5,.6), main="3/5 Average of 5 - Sample Sizes Learning Curve", xaxt="n", type="l", lty=4, xlab="Sample Size (Percent of Training Data)", ylab="accuracy", col="blue")
#axis(1, at=1:10, labels=seq(.01, .1, .01))
#lines(results3_5_LCMAT$mean_test_acc, lty=1, type="l", col="green")
#legend( x="topright", 
#        c("Train", "Test"),
#        col=c("blue", "green"), lty=c(1,1), lwd=4,cex=.5)
```

Comments on the observed learning curves.

The 0/1 learning curve shows underfitting at sample size 127 (.01 of training data) but seems to perform relatively consistent at above 90% accuracy for both testing and training for at all sample sizes.  

The 3/5 learning curve shows underfitting at sample size 116 (.01 of training data) then shows some overfitting at 347 (.03) samples and beyond.  Sample size 231 (.02) appears to be the peak for both testing and training predictions.   
