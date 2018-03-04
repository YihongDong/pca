from readdata import *
from pylab import *
from Bayes import *
from sklearn.model_selection import LeavePOut

def fisher():
    path_boy = "F:\\study in school\\machine learning\\forstudent\\实验数据\\boynew.txt"
    path_girl = "F:\\study in school\\machine learning\\forstudent\\实验数据\\girlnew.txt"
    # height = []
    # weight = []
    # feetsize = []
    x_boy=[]
    x_girl=[]
    label_boy = []  # 1表示男，0表示女
    label_girl = []
    readdata2(path_boy, x_boy, label_boy, 1)
    readdata2(path_girl, x_girl, label_girl, 0)
    x_boy=np.mat(x_boy)
    x_girl=np.mat(x_girl)
    m1=x_boy.mean(0)
    m0=x_girl.mean(0)
    S1=(x_boy-m1[0]).T*(x_boy-m1[0])
    S0=(x_girl-m0[0]).T*(x_girl-m0[0])
    Sw=S1+S0
    S_inverse=Sw.I
    W = S_inverse *(m1-m0).T
    M1=float(W.T*m1.T)
    M0=float(W.T*m0.T)
    w_decision0 =(M0+M1)/2
    path_boy_test = "F:\\study in school\\machine learning\\forstudent\\实验数据\\boy.txt"
    path_girl_test ="F:\\study in school\\machine learning\\forstudent\\实验数据\\girl.txt"
    x = []
    label =[]
    readdata2(path_boy_test, x, label, 1)
    readdata2(path_girl_test, x, label, 0)
    label_test = []
    y = x * W
    errorcount = 0
    for i in range(len(label)):
        if float(y[i] > w_decision0):
            label_test.append(1)
            if label[i] != 1:
                errorcount = errorcount + 1
        else:
            label_test.append(0)
            if label[i] != 0:
                errorcount = errorcount + 1

    e_percentage = errorcount / len(label_test)
    print('fisher测试集的错误率为%f' % e_percentage)


    figure(3)
    FPR, TPR = get_roc_fisher(W, w_decision0, x, label)
    plot(FPR, TPR, label='fisher')



    figure(5)
    x1=np.arange(130,190,0.01)
    y1=(w_decision0-W[0]*x1)/W[1]
    plot(x1,array(y1)[0])
    plot(x1,x1*float(W[1])/float(W[0]))
    for i in range(len(label)):
        if label[i]==1:
            plot(float(x[i][0]),float(x[i][1]),'o',color='r')
        else:
            plot(float(x[i][0]), float(x[i][1]), 'o', color='g')
        a=(float(x[i][1])+float(x[i][0])*float(W[0])/float(W[1]))/\
            (float(W[1])/float(W[0])+float(W[0])/float(W[1]))
        b=a*float(W[1])/float(W[0])
        plot([float(x[i][0]),a],[float(x[i][1]),b],'--',color='0.75')

    axis([140,190,35,85])






