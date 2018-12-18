# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 15:27:04 2018

@author: user
"""
#改进的随机梯度上升算法
import numpy as np
import matplotlib.pyplot as plt
import random


def loadDataSet():
    '''
    加载数据
    '''
    dataMat = []  #创建数据列表
    labelMat = []  #创建标签列表
    fr = open('testSet.txt','r')  #打开文件
    for line in fr.readlines():  #逐行读取数据
        lineArr = line.strip().split()  #去掉其他符号，放入列表中
        dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])])  #添加数据
        labelMat.append(int(lineArr[2]))  #添加标签
    fr.close()  #关闭文件
    return dataMat,labelMat  #返回数据列表和标签列表


def sigmoid(inX):
    '''
    sigmoid函数（logistic函数）
    输入参数：Z
    输出参数：函数结果
    '''
    return 1.0 / (1 + np.exp(-inX))


def stocGradAscent1(dataMatrix,classLabels,numIter = 150):
    '''
    使用随机梯度上升方法对权重进行计算
    '''
    m,n = np.shape(dataMatrix)  #求出数据集的样本个数和特征数
    weights = np.ones(n)  #构造权重矩阵
    for j in range(numIter):  #迭代150次
        dataIndex = list(range(m))  #将样本个数的索引做成列表
        for i in range(m):
            alpha = 4/(1.0+j+i) +0.01  #降低alpha的大小
            randIndex = int(random.uniform(0,len(dataIndex)))  #随机选取一个样本
            h = sigmoid(sum(dataMatrix[randIndex]*weights))  #随机选择一个样本计算函数值
            error = classLabels[randIndex] - h  #计算y-g(ΘX)
            weights = weights + alpha*error*dataMatrix[randIndex]  #更新回归系数
            #del (dataMatrix[randIndex])  #删除已经使用的样本
    return weights


def plotBestFit(weights):
    '''
    绘制数据集和函数
    '''
    dataMat,labelMat = loadDataSet()  #加载数据集
    dataArr = np.array(dataMat)  #转换成numpy的数组形式
    n = np.shape(dataMat)[0]  #求出数据个数
    xcord1 = []  #正样本
    ycord1 = []
    xcord2 = []  #负样本
    ycord2 = []
    for i in range(n):  #根据数据集标签进行分类
        if int(labelMat[i]) == 1:  #如果是正样本
            xcord1.append(dataArr[i,1])
            ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1])
            ycord2.append(dataArr[i,2])
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1,ycord1,color='red',marker='s',alpha=.5)  #绘制正样本
    ax.scatter(xcord2,ycord2,color='green',alpha=.5)  #绘制负样本
    x = np.arange(-3.0,3.0,0.1)
    y = (-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x,y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()
            
        
    if __name__ == '__main__':
    dataMat, labelMat = loadDataSet()
    weights = stocGradAscent1(np.array(dataMat), labelMat)
    print(weights)
    plotBestFit(weights)
    






























    
    
