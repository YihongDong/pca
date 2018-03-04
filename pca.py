from numpy import *
from sklearn.decomposition import PCA
from sklearn.utils.extmath import svd_flip
from pylab import *
from readdata import *
from fisher import *


if __name__=='__main__':
    x = []
    label = []
    path_boy = "F:\\study in school\\machine learning\\forstudent\\实验数据\\boynew.txt"
    path_girl = "F:\\study in school\\machine learning\\forstudent\\实验数据\\girlnew.txt"
    readdata(path_boy, x, label, 1)
    readdata(path_girl, x, label, 0)

    x1=[]
    label1=[]
    readdata(path_boy, x1, label1, 1)
    x0=[]
    label0=[]
    readdata(path_girl, x0, label0, 0)

    x_mean=mat(x).mean(0)
    x1_mean = mat(x1).mean(0)
    x0_mean = mat(x0).mean(0)

    St=(x-x_mean).T*(x-x_mean)/len(label)
    Sw=((x1-x1_mean).T*(x1-x1_mean)+(x0-x0_mean).T*(x0-x0_mean))/len(label)
    Sb=St-Sw
    U,sigma,V_T=linalg.svd(x-x_mean)
    U1, sigma1, V_T1 = linalg.svd(St)
    U2,sigma2,V_T2=linalg.svd(Sw.I*Sb)
    # print(sigma/sigma.sum())
    # print(U.T)
    # pca = PCA(n_components=2)
    # print(V_T)
    # print(V_T1)
    x_pca=(x-x_mean)*V_T.T[:,:2]#array(U)[:,:2]*sigma[:2]
    # U, S, V = pca._fit(x)
    # print(pca.fit_transform(x))

    print(V_T)


    x_lda = (x-x_mean) * V_T2.T[:, :2]
    print(V_T2)

    fisher()


    path_boy_test = "F:\\study in school\\machine learning\\forstudent\\实验数据\\boy.txt"
    path_girl_test ="F:\\study in school\\machine learning\\forstudent\\实验数据\\girl.txt"
    x_test = []
    label_test =[]
    readdata(path_boy_test, x_test, label_test, 1)
    readdata(path_girl_test, x_test, label_test, 0)

    x1=x_pca[:len(x1)]
    x0=x_pca[len(x1):]
    m1 = x1.mean(0)
    m0 = x0.mean(0)
    S1 = (x1 - m1[0]).T * (x1 - m1[0])
    S0 = (x0 - m0[0]).T * (x0 - m0[0])
    Sw = S1 + S0
    S_inverse = Sw.I
    W = S_inverse * (m1 - m0).T
    M1 = float(W.T * m1.T)
    M0 = float(W.T * m0.T)
    w_decision0 = (M0 + M1) / 2
    x_test_pca= (x_test-x_mean)*V_T.T[:,:2]

    y = x_test_pca * W
    errorcount = 0
    for i in range(len(label_test)):
        if float(y[i] > w_decision0):
            if label_test[i] != 1:
                errorcount = errorcount + 1
        else:
            if label_test[i] != 0:
                errorcount = errorcount + 1

    e_percentage = errorcount / len(label_test)
    print('fisher_PCA测试集的错误率为%f' % e_percentage)

    figure(1)
    title("PCA")
    X = np.arange(x_pca[:,0].min(),x_pca[:,0].max(), 0.01)
    Y = (w_decision0 - W[0] * X) / W[1]
    plot(X, array(Y)[0])
    for i in range(len(label)):
        if label[i]==1:
            plot(x_pca[i,0],x_pca[i,1],'o',color='r')
        else:
            plot(x_pca[i,0],x_pca[i,1],'o',color='g')
    figure(1).show()


    figure(3)
    FPR, TPR = get_roc_fisher1(W, w_decision0, x_test_pca, label_test)
    plot(FPR, TPR, label='fisher_pca')

    x1 = x_lda[:len(x1)]
    x0 = x_lda[len(x1):]
    m1 = x1.mean(0)
    m0 = x0.mean(0)
    S1 = (x1 - m1[0]).T * (x1 - m1[0])
    S0 = (x0 - m0[0]).T * (x0 - m0[0])
    Sw = S1 + S0
    S_inverse = Sw.I
    W = S_inverse * (m1 - m0).T
    M1 = float(W.T * m1.T)
    M0 = float(W.T * m0.T)
    w_decision0 = (M0 + M1) / 2
    x_test_lda = (x_test - x_mean) * V_T2.T[:,:2]

    y = x_test_lda * W
    errorcount = 0
    for i in range(len(label_test)):
        if float(y[i] > w_decision0):
            if label_test[i] != 1:
                errorcount = errorcount + 1
        else:
            if label_test[i] != 0:
                errorcount = errorcount + 1

    e_percentage = errorcount / len(label_test)
    print('fisher_LDA测试集的错误率为%f' % e_percentage)

    figure(2)
    title("LDA")
    X = np.arange(x_lda[:,0].min(),x_lda[:,0].max(), 0.01)
    Y = (w_decision0 - W[0] * X) / W[1]
    plot(X, array(Y)[0])
    for i in range(len(label)):
        if label[i]==1:
            plot(x_lda[i,0],x_lda[i,1],'o',color='r')
        else:
            plot(x_lda[i,0],x_lda[i,1],'o',color='g')
    figure(2).show()

    figure(3)
    FPR, TPR = get_roc_fisher1(W, w_decision0, x_test_lda, label_test)
    plot(FPR, TPR, label='fisher_lda')
    plot([0,1],[1,0])
    legend(loc='lower right')

    figure(3).show()
    figure(5).show()

    pass