# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 19:29:53 2018

@author: user
"""

from math import log
#创建数据集
def createDataSet():
    dataSet = [[0, 0, 0, 0, 'no'],         #数据集
            [0, 0, 0, 1, 'no'],
            [0, 1, 0, 1, 'yes'],
            [0, 1, 1, 0, 'yes'],
            [0, 0, 0, 0, 'no'],
            [1, 0, 0, 0, 'no'],
            [1, 0, 0, 1, 'no'],
            [1, 1, 1, 1, 'yes'],
            [1, 0, 1, 2, 'yes'],
            [1, 0, 1, 2, 'yes'],
            [2, 0, 1, 2, 'yes'],
            [2, 0, 1, 1, 'yes'],
            [2, 1, 0, 1, 'yes'],
            [2, 1, 0, 2, 'yes'],
            [2, 0, 0, 0, 'no']]
    labels = ['年龄','有工作','有自己的房子','信贷情况']
    return dataSet, labels 


def calShannonEnt(dataSet):
    '''
    计算香农熵，输入参数dataSet，输出shannonEnt
    '''
    numEntires = len(dataSet)  #计算数据集行数
    labelsCount = {}  #保存每个标签出现的字典
    for featVec in dataSet:  #dataSet是列表里面套着列表，所以用for循环取出里面的子列表
        currentLabel = featVec[-1]  #提取每一个样本里面的标签信息
        #构造字典里面的信息
        if currentLabel not in labelsCount.keys():
            labelsCount[currentLabel] = 0  #没有这个keys就创建一个
        labelsCount[currentLabel] += 1  #计数
        
    shannonEnt = 0.0  #创建香农熵值
    
    for key in labelsCount:
        prob = float(labelsCount[key]) / numEntires
        shannonEnt -= prob * log(prob,2)
        
    return shannonEnt

def splitDataSet(dataSet,axis,value):
    '''
    将数据集划分为你自己想要的子集，输入参数：dataSet，axis,value
    输出参数：retDataSet
    '''
    retDataSet = []  #创建子集列表
    for featVec in dataSet:  #遍历列表，找出每个样本
        if featVec[axis] == value:  #如果找到的标签和你输入的标签一样，那么就要处理一下数据集
            reducedfeatVec = featVec[:axis]  #去掉axis特征
            reducedfeatVec.extend(featVec[axis+1:])  #合并列表
            retDataSet.append(reducedfeatVec)  #划分后的数据集
    return retDataSet

# =============================================================================
# dataSet,labels = createDataSet()
# print(splitDataSet(dataSet,0,0))
# =============================================================================

def chooseBestFeatureToSplit(dataSet):
    '''
    选择最大信息增益的索引值
    输入参数：dataSet 数据集
    输出参数：bestFeature 最大信息增益的索引值
    '''
    numFeatures = len(dataSet[0]) - 1 #求出特征数量
    baseEntry = calShannonEnt(dataSet)  #计算数据集的香农熵
    bestInfoGain = 0.0  #信息增益
    bestFeature = 0  #最大信息增益索引
    for i in range(numFeatures):  #遍历列表里面的所有特征
        featList = [example[i] for example in dataSet]  #获取dataSet第i个特征的所有元素
        uniqueVal = set(featList)  #创建set集合，里面的所有元素不重复
        newEntry = 0.0  #经验条件熵
        for value in uniqueVal:
            subDataSet = splitDataSet(dataSet,i,value)  #划分之后的子集
            prob = len(subDataSet) / float(len(dataSet))  #计算子集的概率
            newEntry += prob*calShannonEnt(subDataSet)  #计算经验条件熵
        infoGain = baseEntry - newEntry  #计算信息增益
        print('第%d个特征的信息增益是%.3f' %(i,infoGain))
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    
    return bestFeature

dataSet,features = createDataSet()
print('最佳索引:' + str(chooseBestFeatureToSplit(dataSet)))
        
        
    
    
    
    

























