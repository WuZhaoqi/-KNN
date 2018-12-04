# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 10:16:29 2018

@author: user
"""

import KNN

group,labels = KNN.createDataSet()
print(group)
print(labels)
# =============================================================================
# print(group.shape[0])
# print(group.shape)
# =============================================================================
def classify0(inX,dataSet,labels,k):
    '''intX是测试用例，即将用来进行分类。dataSet是已经准备好的数据集，
    labels是标签，k是前k个数据'''
#先求出测试用例和数据集之间的距离
    dataSetsize =dataSet.shape[0]  #求出有几个数据
    diffMat = np.tile(inX,(dataSetsize,1)) - dataSet  #重复生成n个数据，减去数据集里的数据
    sqDiffMat = diffMat**2
    sqDistance = sqDiffMat.sum(axis =1)   #求出每个数据的行和
    distance = sqDistance**0.5            #求出最终距离
    #return distance
    #print(classify0([0,2],group,labels))

#选择距离最小的k个点
    sortedDistIndicies = np.argsort(distance)  #对距离进行由小到大排序并且返回索引值
    classcount = {}  #生成一个记录类别次数的字典
    for i in range(k):  #取出前k个点的类别
        voteIlabel = labels[sortedDistIndicies[i]]  #前k个点的标签
        classcount[voteIlabel] = classcount.get(voteIlabel,0) +1  #构造字典
    #print(classcount)
    
#对这些点进行排序
    sortedclasscount = sorted(classcount.items(),key = operator.itemgetter(1),reverse=True)#对字典进行降序排序
    return sortedclasscount[0][0]

print(classify0([0,2],group,labels,3))
