# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 09:41:21 2016

@author: VANLOI
"""

from BaseOneClass import CentroidBasedOneClassClassifier, DensityBasedOneClassClassifier
from sklearn.metrics import roc_curve, auc
from sklearn import svm
import sklearn as sk
from ProcessingData import normalize_data
from LOF import LocalOutlierFactor
import numpy as np
def evaluate(actual, predictions):
    FPR, TPR, thresholds = roc_curve(actual, predictions)
    cen = auc(FPR, TPR) 
    RightIndex=(TPR+(1-FPR)-1)
    index=np.argmax(RightIndex)
    tpr_val=TPR[index]
    fpr_val=FPR[index]
    thresholds_val=thresholds[index]
    y_pred=[0 if i<thresholds_val else 1 for i in predictions]
    y_pred_test=y_pred
    y_test_test=actual
    pre=sk.metrics.precision_score(y_test_test, y_pred_test)
    rec=sk.metrics.recall_score(y_test_test, y_pred_test)
    f1= sk.metrics.f1_score(y_test_test, y_pred_test)
    print("AUC", cen)
    print("Precision", pre)
    print( "Recall",rec)
    print( "f1_score", f1)
    mat=sk.metrics.confusion_matrix(y_test_test, y_pred_test)
    tp=mat[1][1]
    fn=mat[1][0]
    fp=mat[0][1]
    tn=mat[0][0]
    print("TPR",tp/(tp+fn))
    print("FPR",fp/(tn+fp))
    
def auc_density(training_set, testing_set, actual, scale):
    """Compute AUC for density-based methods: Centroid, Negative Mean Distances,
    Kernel Density Estimation and One-class Support Vector Machine, and LOF. 
    """             
    #gamma = 1/2bw^2 = 1/n_feautes -> bw = (n_features/2)^0.5
    #h_default = (d/2.0)**0.5
    bw = (training_set.shape[1]/2.0)**0.5        #default value in One-class SVM
    gamma = 1/(2*bw*bw)
    
    "*************** Centroid AE - Hidden layer **************"
#     CEN = CentroidBasedOneClassClassifier()    
#     CEN.fit(training_set)  
#     predictions_cen = -CEN.get_density(testing_set)
#     FPR_cen, TPR_cen, thresholds_cen = roc_curve(actual, predictions_cen)
#     cen = auc(FPR_cen, TPR_cen)
    CEN = CentroidBasedOneClassClassifier()
    CEN.fit(training_set)
    predictions= -CEN.get_density(testing_set)
    FPR_cen, TPR_cen, thresholds_cen = roc_curve(actual, predictions)
    cen = auc(FPR_cen, TPR_cen)
    print("cen")
    evaluate(actual, predictions)

    

    "****************** Negative Distance - Hidden layer **********************"
#     clf_dis = DensityBasedOneClassClassifier(bandwidth = bw,
#                                              kernel="really_linear",
#                                              metric="euclidean",
#                                              scale = scale)
#     clf_dis.fit(training_set)     
#     predictions_dis  = clf_dis.get_density(testing_set)  
#     FPR_dis, TPR_dis, thresholds_dis = roc_curve(actual, predictions_dis)
#     dis = auc(FPR_dis, TPR_dis)
#     print("Negative Distance/ DensityBasedOneClassClassifier")
#     evaluate(actual, predictions_dis)
    
    "****************** KDE AE - Hidden layer*****************"
    #  ['gaussian'|'tophat'|'epanechnikov'|'exponential'|'linear'|'cosine']
    KDE = DensityBasedOneClassClassifier(bandwidth = bw, 
                                         kernel="gaussian", 
                                         metric="euclidean",
                                         scale = scale)
    KDE.fit(training_set)
    predictions_kde = KDE.get_density(testing_set)
    FPR_kde, TPR_kde, thresholds_kde = roc_curve(actual, predictions_kde)
    kde = auc(FPR_kde, TPR_kde)
    print("KDE AE/DensityBasedOneClassClassifier")
    evaluate(actual, predictions_kde)
    
    "********************* 1-SVM Hidden layer ***************************"
    training_set, testing_set =  normalize_data(training_set, testing_set, scale)
    
    clf_05 = svm.OneClassSVM(nu=0.5, kernel="rbf", gamma=gamma)
    clf_05.fit(training_set)
    #n_support_vectors =  len(clf.support_vectors_)
    predictions_svm  = clf_05.decision_function(testing_set)
    FPR_svm, TPR_svm, thresholds_svm = roc_curve(actual, predictions_svm)
    svm_05 = auc(FPR_svm, TPR_svm)
    print("SVM05")
    evaluate(actual, predictions_svm)
    "nu = 0.1"
    clf_01 = svm.OneClassSVM( nu=0.1, kernel="rbf", gamma=gamma)
    clf_01.fit(training_set)    
    #num_01 =  len(clf_01.support_vectors_) 
    predictions_svm_01  = clf_01.decision_function(testing_set)
    FPR_svm_01, TPR_svm_01, thresholds_svm_01 = roc_curve(actual, predictions_svm_01)
    svm_01 = auc(FPR_svm_01, TPR_svm_01) 
    print("SVM01")
    evaluate(actual, predictions_svm_01)
    "******************************* LOF **********************************"    
    neighbors = (int)(len(training_set)*0.1)
    clf_lof = LocalOutlierFactor(n_neighbors=neighbors)
    clf_lof.fit(training_set)
    predict = clf_lof._decision_function(testing_set)
    FPR, TPR, thresholds = roc_curve(actual, predict)
    lof = auc(FPR, TPR)
    print("LOF")
    evaluate(actual,predict)
    lof=cen=dis= kde=svm_05=svm_01=0
    return lof, cen,dis, kde, svm_05, svm_01


def auc_AEbased(test_X, output_test, actual):
    
    "******************* Testing Output layer ***************"
    OF = -(((test_X - output_test)**2).mean(1))  
    """Classification decision will be based on the error (MAE or RMSE) between
    output and input. The higher error value a example has, the stronger decision 
    the example belongs to anomaly class. 
    Because we set normal class is positive class, so we put minus "-" to MSE to
    make OF of normal examples are large while those from anomaly examples are small
    """
    predictions_auto = OF
    FPR_auto, TPR_auto, thresholds_auto = roc_curve(actual, predictions_auto)
    auc_auto = auc(FPR_auto, TPR_auto)
    
    return auc_auto
    
    
    
    
