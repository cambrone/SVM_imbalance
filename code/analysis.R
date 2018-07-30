###########################################################################################
# TASK: Compare performance of SVM trained using balancing algorithms
# Author: Andres Cambronero
# Project: Comparison of Oversampling Algorithms to Classify Imbalanced Data
# Date Started: July 2, 2018
# Latest Update: July 29, 2018
###########################################################################################

#clear environment
rm(list=ls())

#set working directory
setwd("~/Desktop/summer_projects/Imbalanced Data/data")

#load libraries
library(e1071)
library(smotefamily)
library(cvAUC)
library(ROCR)
library(DMwR)

#load data
imb<-read.csv("imbalanced_train.csv", colClasses = "character")
bal_over<-read.csv("balanced_over_train.csv", colClasses = "character")
bal_under<-read.csv("balanced_under_train.csv", colClasses = "character")
test<-read.csv("test.csv", colClasses = "character")


#change columns class from character to numeric in training and test set
#traning sets
imb[,6:30]<-sapply(imb[,6:30], as.numeric)
bal_over[,6:30]<-sapply(bal_over[,6:30], as.numeric)
bal_under[,6:30]<-sapply(bal_under[,6:30], as.numeric)

#test set
test[,6:30]<-sapply(test[,6:30], as.numeric)

#change class of CHARTER from character to class factor in training and test set 
#traning sets
imb$CHARTER<-as.factor(imb$CHARTER)
bal_over$CHARTER<-as.factor(bal_over$CHARTER)
bal_under$CHARTER<-as.factor(bal_under$CHARTER)

#test set
test$CHARTER<-as.factor(test$CHARTER)

#change class of PROF_LEVEL from character to class factor in training and test set
#traning sets
imb$PROF_LEVEL<-as.factor(imb$PROF_LEVEL)
bal_over$PROF_LEVEL<-as.factor(bal_over$PROF_LEVEL)
bal_under$PROF_LEVEL<-as.factor(bal_under$PROF_LEVEL)

#test set
test$PROF_LEVEL<-as.factor(test$PROF_LEVEL)

#Drop variables that will not be used in training and test set
#traning sets
imb$ENTITY_CD<-NULL
imb$ENTITY_NAME <-NULL    
imb$DISTRICT_NAME<-NULL
imb$COUNTY_NAME<-NULL

bal_over$ENTITY_CD<-NULL
bal_over$ENTITY_NAME <-NULL    
bal_over$DISTRICT_NAME<-NULL
bal_over$COUNTY_NAME<-NULL

bal_under$ENTITY_CD<-NULL
bal_under$ENTITY_NAME <-NULL    
bal_under$DISTRICT_NAME<-NULL
bal_under$COUNTY_NAME<-NULL

#test sets
test$ENTITY_CD<-NULL
test$ENTITY_NAME <-NULL    
test$DISTRICT_NAME<-NULL
test$COUNTY_NAME<-NULL

#Normalize variables in training and test sets 
#training sets
imb[,2:22]<-scale(imb[,2:22])
imb[,24:26]<-scale(imb[,24:26])

bal_over[,2:22]<-scale(bal_over[,2:22])
bal_over[,24:26]<-scale(bal_over[,24:26])

bal_under[,2:22]<-scale(bal_under[,2:22])
bal_under[,24:26]<-scale(bal_under[,24:26])

#test set
test[,2:22]<-scale(test[,2:22])
test[,24:26]<-scale(test[,24:26])

############################################
# ANALYSIS OF PERFORMANCE: IMBALANCED DATA #
############################################
#set seed
set.seed(1)

#train SVM on imbalanced
svm_imb <- svm(PROF_LEVEL ~ ., data = imb, kernel="polynomial", degree=2, cost=5, probability=TRUE, cross=10)

#prediction of observations in test data
imb_pred<-predict(svm_imb, test[,-1], probability=TRUE)

#confusion matrix
imb_confmat <- table(true = test[,1],pred = imb_pred)
write.csv(imb_confmat, "imb_confmat.csv", row.names = T)

#GMEAN
#gmean function
gmean<-function(confmat){
  acc_neg=confmat[1,1]/sum(confmat[1,1]+confmat[1,2])
  acc_pos=confmat[2,2]/sum(confmat[2,2]+confmat[2,1])
  gmean_val=sqrt(acc_neg*acc_pos)
  
  return(gmean_val)
}

#calculate gmean on test data for SVMtrained on imbalanced data
imb_gmean<-gmean(imb_confmat)

#Precision  
#precision function
precision<-function(confmat){
  precis_val=confmat[2,2]/sum(confmat[2,2]+confmat[1,2])
  
  return(precis_val)
}

#calculate precision on test data for SVMtrained on imbalanced data
imb_precision<-precision(imb_confmat)

#Recall
#recall function
recall<-function(confmat){
  recall_val=confmat[2,2]/sum(confmat[2,2]+confmat[2,1])
  
  return(recall_val)
}

#calculate recall on test data for SVMtrained on imbalanced data
imb_recall<-recall(imb_confmat)


#F-MEASURE
#F-measure function 
fmeasure<-function(confmat){
  precis_val=confmat[2,2]/sum(confmat[2,2]+confmat[1,2])
  recall_val=confmat[2,2]/sum(confmat[2,2]+confmat[2,1])
  
  F_val=(2*precis_val*recall_val)/(precis_val+recall_val)
  
  return(F_val)
}

#calculate fmeasure on test data for SVMtrained on imbalanced data
imb_f_measure<-fmeasure(imb_confmat)

#ROC Curve
#plot ROC curve for SVMtrained on imbalanced data
write.csv(attr(imb_pred,"probabilities"), "imb_pred.csv", row.names = F)
svm_rocr<-prediction(attr(imb_pred,"probabilities")[,2], test[,1] == "Proficient")
svm_perf<-performance(svm_rocr, measure = "tpr", x.measure = "fpr")
plot(svm_perf,col="RED")

##calculate AUC on test data for SVMtrained on imbalanced data
imb_auc<-as.numeric(performance(svm_rocr, measure = "auc", x.measure = "cutoff")@ y.values)
write.csv(imb_auc, "imb_auc.csv", row.names = F)


#################################################
# ANALYSIS OF PERFORMANCE: RANDOM OVER-Sampling #
#################################################
#set seed 
set.seed(1)

#train SVM on randomly oversampled data
svm_bal_over <- svm(PROF_LEVEL ~ ., data = bal_over, kernel="polynomial", degree=2, cost=5, probability=TRUE, cross=10)

#prediction observations on test data
bal_over_pred<-predict(svm_bal_over, test[,-1], probability=TRUE)

#confusion matrix
bal_over_confmat <- table(true = test[,1],pred = bal_over_pred)
write.csv(bal_over_confmat, "bal_over_confmat.csv", row.names = T)

##calculate gmean on test data for SVMtrained on randomly oversampled data
bal_over_gmean<-gmean(bal_over_confmat)


#calculate precision on test data for SVMtrained on randomly oversampled data
bal_over_precision<-precision(bal_over_confmat)
  
  
##calculate recall on test data for SVMtrained on randomly oversampled data
bal_over_recall<-recall(bal_over_confmat)

##calculate fmeasure on test data for SVMtrained on randomly oversampled data
bal_over_F<-fmeasure(bal_over_confmat)


##plot ROC on test data for SVMtrained on randomly oversampled data
write.csv(attr(bal_over_pred,"probabilities"), "bal_over_pred.csv", row.names = F)
svm_rocr<-prediction(attr(bal_over_pred,"probabilities")[,1], test[,1] == "Proficient")
svm_perf<-performance(svm_rocr, measure = "tpr", x.measure = "fpr")
plot(svm_perf,add=TRUE, col="BLUE")

##calculate AUC on test data for SVMtrained on randomly oversampled data
bal_over_auc<-as.numeric(performance(svm_rocr, measure = "auc", x.measure = "cutoff")@ y.values)
write.csv(bal_over_auc, "bal_over_auc.csv", row.names = F)



#################################################
# ANAYSIS OF PERFORMANCE: RANDOM UNDER-Sampling #
#################################################
#set seed 
set.seed(1)

#train SVM on randomly under sampling
svm_bal_under <- svm(PROF_LEVEL ~ ., data = bal_under, kernel="polynomial", degree=2, cost=5, probability=TRUE, cross=10)

#prediction observations on test data
bal_under_pred<-predict(svm_bal_under, test[,-1], probability=TRUE)

#confusion matrix
bal_under_confmat <- table(true = test[,1],pred = bal_under_pred)
write.csv(bal_under_confmat, "bal_under_confmat.csv", row.names = T)


##calculate gmean on test data for SVM trained on randomly undersampled data
bal_under_gmean<-gmean(bal_under_confmat)


###calculate precision on test data for SVM trained on randomly undersampled data  
bal_under_precision<-precision(bal_under_confmat)


##calculate recall on test data for SVM trained on randomly undersampled data
bal_under_recall<-recall(bal_under_confmat)

##calculate fmeasure on test data for SVM trained on randomly undersampled data
bal_under_F<-fmeasure(bal_under_confmat)


##plot ROC on test data for SVM trained on randomly undersampled data
write.csv(attr(bal_under_pred,"probabilities"), "bal_under_pred.csv", row.names = F)
svm_rocr<-prediction(attr(bal_under_pred,"probabilities")[,1], test[,1] == "Proficient")
svm_perf<-performance(svm_rocr, measure = "tpr", x.measure = "fpr")
plot(svm_perf, add=TRUE, col="GREEN")

##calculate AUC on test data for SVMtrained on randomly undersampled data
bal_under_auc<-as.numeric(performance(svm_rocr, measure = "auc", x.measure = "cutoff")@ y.values)
write.csv(bal_under_auc, "bal_under_auc.csv", row.names = F)


#####################################
#ANALYSIS OF PERFORMANCE: SMOTE
#####################################
#create balanced dataset with synthetic data points
smote_data<-DMwR::SMOTE(PROF_LEVEL ~., imb, perc.over = 1000, perc.under = 110 , k=5)

#set seed
set.seed(1)

#train SVM using smote data
svm_smote <- svm(PROF_LEVEL ~ ., data = smote_data, kernel="polynomial", degree=2, cost=5, probability=TRUE, cross=10)

#predict test observations
smote_pred<-predict(svm_smote, test[,-1], probability=TRUE)

#confustion matrix
smote_confmat <- table(true = test[,1],pred = smote_pred)
write.csv(smote_confmat, "smote_confmat.csv", row.names = T)

##calculate gmean on test data for SVM trained on SMOTE
smote_gmean<-gmean(smote_confmat)


##calculate precision on test data for SVM trained on SMOTE
smote_precision<-precision(smote_confmat)


##calculate recall on test data for SVM trained on SMOTE
smote_recall<-recall(smote_confmat)

##calculate fmeasure on test data for SVM trained on SMOTE
smote_F<-fmeasure(smote_confmat)


##plot ROC for SVM trained on SMOTE
write.csv(attr(smote_pred,"probabilities"), "smote_pred.csv", row.names = F)
svm_rocr<-prediction(attr(smote_pred,"probabilities")[,2], test[,1] == "Proficient")
svm_perf<-performance(svm_rocr, measure = "tpr", x.measure = "fpr")
plot(svm_perf,add=TRUE,col="PURPLE")

##calculate AUC on test data for SVM trained on SMOTE
smote_auc<-as.numeric(performance(svm_rocr, measure = "auc", x.measure = "cutoff")@ y.values)
write.csv(smote_auc, "smote_auc.csv", row.names = F)



################################################
#ANALYSIS OF PERFORMANCE: BORDERLINE SMOTE
################################################

#smotefamily needs all data to be numeric
#change PROF_LEVEL to numeric
imb$PROF_LEVEL<-as.character(imb$PROF_LEVEL)
imb$PROF_LEVEL<-ifelse(imb$PROF_LEVEL=="Not Proficient",0,1)


#change CHARTER to numertic
imb$CHARTER<-as.character(imb$CHARTER)
imb$CHARTER<-as.numeric(imb$CHARTER)


#create balanced boderline smote data
border_smote_data<-BLSMOTE(imb,imb$PROF_LEVEL,K=5,C=4,dupSize=0,method =c("type1"))


#extract data with synthetic data points and drop extra column
border_smote_data<-border_smote_data$data
border_smote_data$class<-NULL


#change PROF_LEVEL to factor to train model
border_smote_data$PROF_LEVEL<-ifelse(border_smote_data$PROF_LEVEL==0,"Not Proficient","Proficient")
border_smote_data$PROF_LEVEL<-as.factor(border_smote_data$PROF_LEVEL)


#synthetic data gives CHARTER values between 0 and 1
#changing to factor would create incorrect levels.
# treat charter as numeric
#border_smote_data$CHARTER<-as.character(border_smote_data$CHATER)
#border_smote_data$CHARTER<-as.factor(border_smote_data$CHATER)



#train SVM on borderline smote data
svm_border <- svm(as.factor(PROF_LEVEL) ~ ., data = border_smote_data, kernel="polynomial", degree=2, cost=5, probability=TRUE, cross=10)

#change CHARTER to numeric in test data
test$CHARTER<-as.character(test$CHARTER)
test$CHARTER<-as.numeric(test$CHARTER)


#predict test observations
border_pred<-predict(svm_border, test[,-1], probability = T)


#confustion matrix
border_confmat <- table(true = test[,1],pred = border_pred)
write.csv(border_confmat, "border_confmat.csv", row.names = T)

##calculate gmean on test data for SVM trained on borderline SMOTE
border_gmean<-gmean(border_confmat)


##calculate precision on test data for SVM trained on borderline SMOTE
border_precision<-precision(border_confmat)


##calculate recall on test data for SVM trained on borderline SMOTE
border_recall<-recall(border_confmat)

##calculate fmeasure on test data for SVM trained on borderline SMOTE
border_F<-fmeasure(border_confmat)

# plot ROC for SVM trained on borderline SMOTE
write.csv(attr(border_pred,"probabilities"), "border_pred.csv", row.names = F)
svm_rocr<-prediction(attr(border_pred,"probabilities")[,1], test[,1] == "Proficient")
svm_perf<-performance(svm_rocr, measure = "tpr", x.measure = "fpr")
plot(svm_perf,add=TRUE,col="BROWN")

##calculate AUC on test data for SVM trained on borderline SMOTE
border_auc<-as.numeric(performance(svm_rocr, measure = "auc", x.measure = "cutoff")@ y.values)
write.csv(border_auc, "border_auc.csv", row.names = F)


###################################
#ANALYSIS OF PERFORMANCE: ADASYN
###################################
#create balanced ADASYN data
adasyn_data<-ADAS(imb,imb$PROF_LEVEL,K=5)


#extract data with synthetic observations
adasyn_data<-adasyn_data$data


#drop extract column
adasyn_data$class<-NULL


#change PROF_LEVEL to factor
adasyn_data$PROF_LEVEL<-ifelse(adasyn_data$PROF_LEVEL==0,"Not Proficient","Proficient")
adasyn_data$PROF_LEVEL<-as.factor(adasyn_data$PROF_LEVEL)


#train model on ADASYN data
svm_adasyn <- svm(as.factor(PROF_LEVEL) ~ ., data = adasyn_data, kernel="polynomial", degree=2, cost=5, probability=TRUE, cross=10)


#predict test observations
adasyn_pred<-predict(svm_adasyn, test[,-1], probability = T)


#confusion matrix
adasyn_confmat <- table(true = test[,1],pred = adasyn_pred)
write.csv(adasyn_confmat, "adasyn_confmat.csv", row.names = T)


##calculate gmean on test data for SVM trained on ADASYN
adasyn_gmean<-gmean(adasyn_confmat)


##calculate precision on test data for SVM trained on ADASYN
adasyn_precision<-precision(adasyn_confmat)


##calculate recall on test data for SVM trained on ADASYN
adasyn_recall<-recall(adasyn_confmat)


##calculate fmeasure on test data for SVM trained on ADASYN
adasyn_F<-fmeasure(adasyn_confmat)



##plot ROC for SVM trained on ADASYN
write.csv(attr(adasyn_pred,"probabilities"), "adasyn_pred.csv", row.names = F)
svm_rocr<-prediction(attr(adasyn_pred,"probabilities")[,1], test[,1] == "Proficient")
svm_perf<-performance(svm_rocr, measure = "tpr", x.measure = "fpr")
plot(svm_perf,add=TRUE,col="black")


##calculate AUC on test data for SVM trained on ADASYN
adasyn_auc<-as.numeric(performance(svm_rocr, measure = "auc", x.measure = "cutoff")@ y.values)
write.csv(adasyn_auc, "adasyn_auc.csv", row.names = F)



#########################################
# ANALYSIS OF PERFORMANCE: SAFE-LEVEL 
#########################################
#Safe Level SMOTE
sl_data<-SLS(imb,imb$PROF_LEVEL,K=5, C=4, dupSize = 0)


#extract data with synthetic data points
sl_data<-sl_data$data
sl_data$class<-NULL


#change PROF_LEVEL to factor
sl_data$PROF_LEVEL<-ifelse(sl_data$PROF_LEVEL==0,"Not Proficient","Proficient")
sl_data$PROF_LEVEL<-as.factor(sl_data$PROF_LEVEL)


#train model on safe-level data
svm_sl <- svm(as.factor(PROF_LEVEL) ~ ., data = sl_data, kernel="polynomial", degree=2, cost=5, probability=TRUE, cross=10)


#predict observations from test data
sl_pred<-predict(svm_sl, test[,-1], probability = T)


#confusion matrix
sl_confmat <- table(true = test[,1],pred = sl_pred)
write.csv(sl_confmat, "sl_confmat.csv", row.names = T)


##calculate gmean on test data for SVM trained on safelevel
sl_gmean<-gmean(sl_confmat)

##calculate precision on test data for SVM trained on safelevel
sl_precision<-precision(sl_confmat)


##calculate precision on test data for SVM trained on safelevel
sl_recall<-recall(sl_confmat)


##calculate fmeasure on test data for SVM trained on safelevel
sl_F<-fmeasure(sl_confmat)


##Plot ROC for SVM trained on safelevel
write.csv(attr(sl_pred,"probabilities"), "sl_pred.csv", row.names = F)
svm_rocr<-prediction(attr(sl_pred,"probabilities")[,1], test[,1] == "Proficient")
svm_perf<-performance(svm_rocr, measure = "tpr", x.measure = "fpr")
plot(svm_perf,add=TRUE,col="yellow")

##calculate AUC on test data for SVM trained on safelevel
sl_auc<-as.numeric(performance(svm_rocr, measure = "auc", x.measure = "cutoff")@ y.values)
write.csv(sl_auc, "sl_auc.csv", row.names = F)




####################################
# COMPARISON OF METRICS
####################################
#ADD LEGEND TO PLOT
legend("bottomright", legend=c("Original", "Rand. Oversamp.",
                          "Rand. Under", "SMOTE", "Borderline",
                          "ADASYN", "Safe-Level"),
       col=c("RED", "BLUE", "GREEN", "PURPLE", 
             "BROWN", "BLACK", "YELLOW"),cex=0.8, lty=1)

#vector of gmeans
gmeans<-c(imb_gmean, bal_over_gmean, 
          bal_under_gmean, smote_gmean, 
          border_gmean, adasyn_gmean, sl_gmean)
          
#labels
methods<-c("Original",  "Rand. Oversamp.",
           "Rand. Under", "SMOTE", "Borderline",
           "ADASYN", "Safe-Level")

#plot gmeans for all methods
gmean_plot<-barplot(gmeans, ylim = c(0,1), ylab = "G-Mean")
axis(1, at=gmean_plot, labels=methods, tick=FALSE,
     las=2, line=-0.5, cex.axis=0.7)
text(x = gmean_plot, y = gmeans, label = round(gmeans,3), pos = 3,
     cex = 0.7, col = c("black","red","black",
                        "black","black","black",
                        "black"))
                                                                                     

#vector of precisions
precisions<-c(imb_precision, bal_over_precision,
              bal_under_precision, smote_precision,
              border_precision, adasyn_precision, sl_precision)
              
#plot precision for all methods              
precision_plot<-barplot(precisions, ylim = c(0,1), ylab = "Precision")
axis(1, at=precision_plot, labels=methods,
     tick=FALSE, las=2, line=-0.5, cex.axis=0.7)
text(x = precision_plot, y = precisions, label = round(precisions,3),
     pos = 3,cex = 0.7, col = c("red", "black","black",
                                "black","black", "black",
                                "black"))
                                                                                      

#vector of recall values
recalls<-c(imb_recall, bal_over_recall,
              bal_under_recall, smote_recall,
              border_recall, adasyn_recall, sl_recall)

#plot of recall
recall_plot<-barplot(recalls, ylim = c(0,1.1), ylab = "Recall")
axis(1, at=recall_plot, labels=methods,
     tick=FALSE, las=2, line=-0.5, cex.axis=0.7)
text(x = recall_plot, y = recalls, label = round(recalls,3),
     pos = 3,cex = 0.7, col = c("black","red","black",
                                "black","black", "black",
                                "black"))

#Vector of fmeasures
Fs<-c(imb_f_measure, bal_over_F,
      bal_under_F, smote_F,
      border_F, adasyn_F, sl_F)
      
#plot of fmeasures
F_plot<-barplot(Fs, ylim = c(0,0.8), ylab = "F-Measure")
axis(1, at=F_plot, labels=methods,
     tick=FALSE, las=2, line=-0.5, cex.axis=0.7)
text(x = F_plot, y = Fs, label = round(Fs,3),
     pos = 3,cex = 0.7, col = c("black","black",
                                "black","red","black", "black",
                                "black"))

#vector of AUCs
AUCs<-c(imb_auc, bal_over_auc, bal_under_auc,
        smote_auc, border_auc, adasyn_auc,sl_auc)
        
#plot of AUCs
AUC_plot<-barplot(AUCs, ylim = c(0,1.2), ylab = "AUC")
axis(1, at=AUC_plot, labels=methods,
     tick=FALSE, las=2, line=-0.5, cex.axis=0.7)
text(x = AUC_plot, y = AUCs, label = round(AUCs,3),
     pos = 3,cex = 0.7, col = c("black","red","black",
                                "black","black", "black",
                                "black"))

