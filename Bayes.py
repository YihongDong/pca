from math import pow,pi,sqrt,exp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pylab import *
from readdata import *
from Normal_distribution import *
from roc_auc import *

theta_boy = 0.5
theta_girl = 1-float(theta_boy)

def Bayes():
    path_boy ="F:\\study in school\\machine learning\\forstudent\\实验数据\\boynew.txt"
    path_girl ="F:\\study in school\\machine learning\\forstudent\\实验数据\\girlnew.txt"
    height = []
    weight = []
    feetsize = []
    label = []  # 1表示男，0表示女
    readdata1(path_boy,height,weight,feetsize,label,1)
    readdata1(path_girl,height,weight,feetsize,label,0)
    #正态分布+极大似然估计
    boy_height_mean,boy_height_variance=onefeature(height,label,1)
    boy_weight_mean,boy_weight_variance=onefeature(weight,label,1)
    boy_feetsize_mean,boy_feetsize_variance=onefeature(feetsize,label,1)
    girl_height_mean,girl_height_variance=onefeature(height,label,0)
    girl_weight_mean,girl_weight_variance=onefeature(weight,label,0)
    girl_feetsize_mean,girl_feetsize_variance=onefeature(feetsize,label,0)

    path_boy_test = "F:\\study in school\\machine learning\\forstudent\\实验数据\\boy.txt"
    path_girl_test ="F:\\study in school\\machine learning\\forstudent\\实验数据\\girl.txt"
    height_test = []
    weight_test = []
    feetsize_test = []
    label_test = []  # 1表示男，0表示女
    label_result= []
    readdata1(path_boy_test,height_test,weight_test,feetsize_test,label_test,1)
    readdata1(path_girl_test, height_test, weight_test, feetsize_test, label_test, 0)

    #双特征
    boy_mean=[boy_height_mean,boy_weight_mean]
    boy_variance=[boy_height_variance,boy_weight_variance]
    girl_mean=[girl_height_mean,girl_weight_mean]
    girl_variance=[girl_height_mean,girl_weight_variance]
    test=[height_test,weight_test]
    label_result = get_result_two(boy_mean, boy_variance, girl_mean, girl_variance,test,
                              label_test, theta_boy)

    risk=array([[0,6],[1,0]])
    label_result = get_result_two_risk(boy_mean, boy_variance, girl_mean, girl_variance, test,
                                  label_test, theta_boy,risk)
    e3 = get_error_percent(label_test, label_result)

    #roc
    figure(3)
    FPR, TPR = get_roc(boy_mean, boy_variance, girl_mean, girl_variance,
                                  test, label_test, 0.5)
    plot(FPR, TPR,label='0.5_two')


    plot([0,1],[1,0])
    legend(loc='lower right')
    figure(3).show()

    #决策面

    fig=figure(5)
    x=np.arange(140,190,1)
    y=np.arange(35,85,1)
    x, y = np.meshgrid(x, y)
    f=(x-boy_height_mean)**2/boy_height_variance+(y-boy_weight_mean)**2/boy_weight_variance- \
      (x - girl_height_mean) ** 2 / girl_height_variance-(y-girl_weight_mean)**2/girl_weight_variance- \
      2 * log(sqrt(girl_height_variance * girl_weight_variance / (boy_weight_variance * boy_height_variance)))
    plt.contour(x, y, f,0)
    show()
