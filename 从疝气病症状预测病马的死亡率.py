# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 19:46:22 2018

@author: user
"""
#当数据集较小时，我们使用梯度上升算法；当数据集较大时，我们使用改进的随机梯度上升算法
import numpy as np
import random


def sigmoid(inX):
    '''
    求出函数值
    '''
    return 1.0/(1 + np.exp(-inX))


def gradAscent(dataMatIn,labelClass):
    '''
    梯度上升算法，用来计算函数的权重
    '''
    dataMatrix = np.mat(dataMatIn)  #将数据集转化为矩阵的形式
    labelMat = np.mat(labelClass).transpose()  #将标签转化为矩阵的形式并且转置
    m,n = np.shape(dataMatrix)  #求出数据集的样本个数和特征个数
    alpha = 0.01  #学习速率
    maxCycles = 500  #迭代次数
    weights = np.ones((n,1))  #构建权重矩阵
    for k in range(maxCycles):  #迭代500次
        h = sigmoid(dataMatrix*weights)  #dataMatrix是m行n列，weights是n行1列，n代表特征的个数
        error = dataMatrix - h  #计算的是y-g(ΘX)
        weights = weights + alpha*dataMatrix.transpose()*error
    return weights.getA()  #将矩阵转换为数组并返回


def classifyVector(inX,weights):
    '''
    分类函数
    输入参数：inX：特征向量，weights：权重
    '''
    prob = sigmoid(sum(inX*weights))  #计算函数的值
    if prob > 0.5:
        return 1.0
    else:
        return 0.0
    
    
def colicTest():
    '''
    使用logistic分类器做测试，测试算法正确率
    '''
    frTrain = open('horseColicTraining.txt')  #打开训练集
    frTest = open('horseColicTest.txt')  #打开测试集
    trainingSet = []
    trainingLabels = []
    for line in frTrain.readlines():  #读取训练集里每个样本的具体信息
        currLine = line.strip().split('\t')  #切分为列表形式
        lineArr = []
        for i in range(len(currLine)-1):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[-1]))
    
    trainWeights = gradAscent(np.array(trainingSet),trainingLabels)  #计算出权重
    errorCount = 0
    numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')  #切分为列表形式
        lineArr = []
        for i in range(len(currLine)-1):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(np.array(lineArr),trainWeights[:,0])) != int(currLine[-1]):
            errorCount += 1
            
    errorRate = (float(errorCount)/numTestVec) *100
    print('测试集错误率：%.2f%%' %errorRate)
    
colicTest()
    

        
            
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    




























    
