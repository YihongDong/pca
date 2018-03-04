import numpy as np
import matplotlib.pyplot as plt
from Normal_distribution import *

def get_roc(boy_mean, boy_variance, girl_mean, girl_variance,
                                  test, label_test, theta_boy):
    FPR = [1]
    TPR = [1]
    for i in np.arange(0.01,0.99,0.01):
        if type(test[1])!=list:
            label_result = get_result(boy_mean, boy_variance, girl_mean, girl_variance,
                                 test, label_test, theta_boy, i)
        else:
            label_result =get_result_two(boy_mean, boy_variance, girl_mean, girl_variance,
                                 test, label_test, theta_boy, i)
        fpr,tpr = get_fpr_tpr(label_test, label_result)
        FPR.append(fpr)
        TPR.append(tpr)
    FPR.append(0)
    TPR.append(0)
    return FPR,TPR



def get_roc_risk(boy_mean, boy_variance, girl_mean, girl_variance,
                                  test, label_test, theta_boy,risk):
    FPR = [1]
    TPR = [1]
    for i in np.arange(0.01,0.99,0.01):
        if type(test[1])!=list:
            label_result = get_result(boy_mean, boy_variance, girl_mean, girl_variance,
                                 test, label_test, theta_boy, i)
        else:
            label_result =get_result_two_risk(boy_mean, boy_variance, girl_mean, girl_variance,
                                 test, label_test, theta_boy,risk,i)
        fpr,tpr = get_fpr_tpr(label_test, label_result)
        FPR.append(fpr)
        TPR.append(tpr)
    FPR.append(0)
    TPR.append(0)
    return FPR,TPR

def get_roc_fisher(w,w0,test,label_test):
    FPR = [1]
    TPR = [1]
    for i in np.arange(-2,2,0.01):
        label_result = get_result_fisher(w,w0,test,i)
        fpr,tpr = get_fpr_tpr(label_test, label_result)
        FPR.append(fpr)
        TPR.append(tpr)
    FPR.append(0)
    TPR.append(0)
    return FPR,TPR

def get_roc_fisher1(w,w0,test,label_test):
    FPR = [0]
    TPR = [0]
    for i in np.arange(-5,5,0.01):
        label_result = get_result_fisher(w,w0,test,i)
        fpr,tpr = get_fpr_tpr(label_test, label_result)
        FPR.append(fpr)
        TPR.append(tpr)
    FPR.append(1)
    TPR.append(1)
    return FPR,TPR