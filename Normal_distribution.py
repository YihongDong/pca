from math import pow,pi,sqrt,exp
import numpy as np

def onefeature(feature,label,sex=1):
    feature_total = 0
    total = 0
    feature_Variance_total = 0
    for i in range(len(label)):
        if label[i]==sex:
            feature_total = feature_total+float(feature[i])
            total = total+1
    feature_mean =feature_total/total
    for i in range(len(label)):
        if label[i]==sex:
            feature_Variance_total = feature_Variance_total +pow((float(feature[i])-feature_mean),2)
    feature_Variance =feature_Variance_total/(total-1)

    return feature_mean,feature_Variance


def probability_density(mean,variance,feature,i):
    p=1 / sqrt(2 * pi * variance) * exp(-((float(feature[i]) - mean)**2) / (2 * variance))
    return p


def get_result(boy_feature_mean,boy_feature_variance,girl_feature_mean,girl_feature_variance,
               feature_test,label,theta_boy,h=0.5):
    label_result=[]
    theta=float(theta_boy)
    for i in range(len(label)):
        p1= probability_density(boy_feature_mean,boy_feature_variance,feature_test,i)
        p0= probability_density(girl_feature_mean,girl_feature_variance,feature_test,i)
        if p1*theta/(p0*(1-theta)+p1*theta)>=h:
            label_result.append(1)
        else:
            label_result.append(0)
    return label_result

def get_error_percent(label_test,label_result):
    error = 0
    for i in range(len(label_test)):
        if label_test[i] != label_result[i]:
            error = error + 1
            # print(i)
    e = error / len(label_test)
    return e

def get_result_two(boy_feature_mean,boy_feature_variance,girl_feature_mean,girl_feature_variance,
                   feature_test,label,theta_boy,h=0.5):
    label_result=[]
    theta=float(theta_boy)
    for i in range(len(label)):
        p1= probability_density(boy_feature_mean[0],boy_feature_variance[0],feature_test[0],i)\
            *probability_density(boy_feature_mean[1],boy_feature_variance[1],feature_test[1],i)
        p0= probability_density(girl_feature_mean[0],girl_feature_variance[0],feature_test[0],i)\
            *probability_density(girl_feature_mean[1],girl_feature_variance[1],feature_test[1],i)
        if p1*theta/(p0*(1-theta)+p1*theta)>=h:
            label_result.append(1)
        else:
            label_result.append(0)
    return label_result

def get_fpr_tpr(label_test,label_result):
    fp = 0
    tp = 0
    fn = 0
    tn = 0
    for i in range(len(label_test)):
        if label_test[i]==1 and label_result[i]==1:
            tp=tp+1
        elif label_test[i]==0 and label_result[i]==0:
            tn=tn+1
        elif label_test[i]==0 and label_result[i]==1:
            fp=fp+1
        else: fn=fn+1

    fpr = fp/(fp+tn)
    tpr = tp/(tp+fn)
    return fpr,tpr

def get_result_two_risk(boy_feature_mean,boy_feature_variance,girl_feature_mean,girl_feature_variance,
                   feature_test,label,theta_boy,risk,h=0.5):
    label_result=[]
    theta=float(theta_boy)
    for i in range(len(label)):
        p1= probability_density(boy_feature_mean[0],boy_feature_variance[0],feature_test[0],i)\
            *probability_density(boy_feature_mean[1],boy_feature_variance[1],feature_test[1],i)
        p0= probability_density(girl_feature_mean[0],girl_feature_variance[0],feature_test[0],i)\
            *probability_density(girl_feature_mean[1],girl_feature_variance[1],feature_test[1],i)
        if p1*theta*risk[0][1]/(p0*(1-theta)*risk[1][0]+p1*theta*risk[0][1])>=h:
            label_result.append(1)
        else:
            label_result.append(0)
    return label_result

def get_result_fisher(w,w0,feature_test,h=1):
    label_result=[]
    for i in range(len(feature_test)):
        if feature_test[i]*w>=h*w0:
            label_result.append(1)
        else:
            label_result.append(0)
    return label_result