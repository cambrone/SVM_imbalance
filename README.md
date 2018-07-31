# SVM_imbalance
SVM performance on imbalance data and SMOTE



This repository contains the code and data to reproduce my report " Comparison of Oversampling Algorithms to Classify Imbalanced Data."  Imbalanced data is a common characteristic of data sets in almost every field.  Most classifiers produce poor test predictions when trained on imbalanced data. A number of preprocessing algorithms exist to combat this problem. This paper explores how SVM performs when trained using several of these data preprocessing algorithms. 

The results suggest that the performance of the SVM trained on different preprocessing techniques is highly dependent on the metric used to assess it. In general, SMOTE and its variants did not perform substantially better in the testing phase than random oversampling, but all preprocessing techniques produced better test results than the SVM trained on imbalanced data. 

The project used data from elementary and middle schools in the state of New York. The data was retrieved from public databases, which are specified in the paper. This report was completed entirely in R.

The project covers the theoretical concepts and applications of the following topics:

-	SVM
-	Random Oversampling
-	Random Undersampling
-	Synthetic Minority Oversampling Technique (SMOTE)
-	Adaptive Synthetic Sampling Approach 
-	Safe-Level SMOTE
-	Borderline SMOTE
