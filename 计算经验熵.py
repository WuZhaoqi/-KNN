# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 18:22:47 2018

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


if __name__ == '__main__':
    dataset,labels = createDataSet()
    print(dataset)
    print(calShannonEnt(dataset))
    
    
    
    
    
            
        
