# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 09:37:41 2018

@author: user
"""
# =============================================================================
# KNN算法完整流程：
# 1、收集数据：使用爬虫获取数据或者已经保存在本地的数据集
# 2、准备数据：对数据进行解析，导入
# 3、处理数据：对数据进行可视化
# 4、测试算法：测试算法的错误率是否在可接受范围之内
# 5、使用算法：正式使用算法
# =============================================================================
import numpy as np


def file2matrix(filename):
    """
    函数说明：打开并解析文件。对文件里的数据进行分类，1代表不喜欢，2代表一般喜欢，3代表很喜欢
    """
    #打开文件
    fr = open(filename)
    #读取文件所有内容
    array0Lines = fr.readlines()
    #文件行数
    numberOfLines = len(array0Lines)
    #构建特征矩阵
    returnMat = np.zeros((numberOfLines,3))
    #构建分类标签向量
    classLabelvector = []
    #print(array0Lines)
#file2matrix('datingTestSet.txt')
    #对读取到的文件进行预处理
    index = 0  #行的索引值
    for line in array0Lines:
        line = line.strip()  #去除空白字符，如\t,\r,\n
        Listfromline = line.split('\t')  #对字符串以\t进行切片
        returnMat[index,:] = Listfromline[0:3]
        #根据文本中标记的喜欢的程度进行分类,1代表不喜欢,2代表魅力一般,3代表极具魅力
        if Listfromline[-1] == 'didntLike':
            classLabelvector.append(1)
        elif Listfromline[-1] == 'smallDoses':
            classLabelvector.append(2)
        elif Listfromline[-1] == 'largeDoses':
            classLabelvector.append(3)
        index += 1
    return returnMat, classLabelvector
#file2matrix('datingTestSet.txt')
    
# =============================================================================
# if __name__ == '__main__':
#     #打开的文件名
#     filename = "datingTestSet.txt"
#     #打开并处理数据
#     datingDataMat, datingLabels = file2matrix(filename)
#     print(datingDataMat)
#     print(datingLabels)
# =============================================================================
    
from matplotlib.font_manager import FontProperties
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np


def showdatas(datingDataMat,datingLabels):
    '''
    对处理过的数据进行可视化，方便进行分析
    参数：datingDataMat 特征矩阵
          datingLabels  分类标签
    '''
    #设置中文格式
    font = FontProperties(fname = r'c:\windows\fonts\simsun.ttc',size = 14)  
    
    #将fig画布划分成四块，2行2列，nrow=2,ncols=2,sharex=False,sharey=False,figsize=(13,8)
    fig,axs = plt.subplots(nrows=2,ncols=2,sharex=False,sharey=False,figsize=(13,8))  #figsize是整个画布的尺寸大小
    #fig是画布，axs才是真正的图表
    #构建标签颜色列表
    LabelsColors =[]
    for i in datingLabels:
        if i == 1:
            LabelsColors.append('black')
        elif i == 2:
            LabelsColors.append('orange')
        elif i == 3:
            LabelsColors.append('red')
    #画出散点图,以datingDataMat矩阵的第一(飞行常客例程)、第二列(玩游戏)数据画散点数据,散点大小为15,透明度为0.5
    axs[0][0].scatter(x=datingDataMat[:,0],y=datingDataMat[:,1],c=LabelsColors,s=15,alpha=0.5)
    #设置图表题目，x轴标签和y轴标签
    axs0_title_text = axs[0][0].set_title(u'每年获得的飞行常客里程数和玩视频游戏所消耗时间百分比',FontProperties=font)
    axs0_xlabel = axs[0][0].set_xlabel(u'每年获得的飞行常客里程数',FontProperties=font)
    axs0_ylabel = axs[0][0].set_ylabel(u'玩视频游戏所消耗时间百分比',FontProperties=font)
    #将题目和标签应用到图表上
    plt.setp(axs0_title_text,size=9,weight='bold',color='red')
    plt.setp(axs0_xlabel,size=7,weight='bold',color='black')
    plt.setp(axs0_ylabel,size=7,weight='bold',color='black')
    #画出散点图,以datingDataMat矩阵的第一(飞行常客例程)、第三列(冰激凌)数据画散点数据,散点大小为15,透明度为0.5
    axs[0][1].scatter(x=datingDataMat[:,0], y=datingDataMat[:,2],s=15, c=LabelsColors, alpha=.5)
    #设置标题,x轴label,y轴label
    axs1_title_text = axs[0][1].set_title(u'每年获得的飞行常客里程数与每周消费的冰激淋公升数',FontProperties=font)
    axs1_xlabel_text = axs[0][1].set_xlabel(u'每年获得的飞行常客里程数',FontProperties=font)
    axs1_ylabel_text = axs[0][1].set_ylabel(u'每周消费的冰激淋公升数',FontProperties=font)
    plt.setp(axs1_title_text, size=9, weight='bold', color='red') 
    plt.setp(axs1_xlabel_text, size=7, weight='bold', color='black') 
    plt.setp(axs1_ylabel_text, size=7, weight='bold', color='black')

    #画出散点图,以datingDataMat矩阵的第二(玩游戏)、第三列(冰激凌)数据画散点数据,散点大小为15,透明度为0.5
    axs[1][0].scatter(x=datingDataMat[:,1], y=datingDataMat[:,2], s=15,c=LabelsColors, alpha=.5)
    #设置标题,x轴label,y轴label
    axs2_title_text = axs[1][0].set_title(u'玩视频游戏所消耗时间占比与每周消费的冰激淋公升数',FontProperties=font)
    axs2_xlabel_text = axs[1][0].set_xlabel(u'玩视频游戏所消耗时间占比',FontProperties=font)
    axs2_ylabel_text = axs[1][0].set_ylabel(u'每周消费的冰激淋公升数',FontProperties=font)
    plt.setp(axs2_title_text, size=9, weight='bold', color='red') 
    plt.setp(axs2_xlabel_text, size=7, weight='bold', color='black') 
    plt.setp(axs2_ylabel_text, size=7, weight='bold', color='black')

    #设置图例
    didntLike = mlines.Line2D([], [], color='black', marker='.',markersize=6, label='didntLike')
    smallDoses = mlines.Line2D([],[],color='orange',marker='.',markersize=6,label='smallDoses')
    largeDoses = mlines.Line2D([],[],color='red',marker='.',markersize=6,label='largeDoses')
    #应用图例
    axs[0][0].legend(handles=[didntLike,smallDoses,largeDoses])
    axs[0][1].legend(handles=[didntLike,smallDoses,largeDoses])
    axs[1][0].legend(handles=[didntLike,smallDoses,largeDoses])
    plt.show()

import numpy as np

   
def autoNorm(dataSet):
    '''
    对数据进行归一化处理
    参数：dataSet
    返回值：normDataSet - 归一化后的特征矩阵
            value_range - 数据范围
            min_value - 数据最小值
    '''
    #求出数据集里面的最大最小值
    max_value = dataSet.max(axis=0)  #第一二三行都求列的大小，所以求每列最大
    min_value = dataSet.min(axis=0)
    #最大值-最小值的范围
    value_range = max_value-min_value
    #构建归一化后的特征矩阵
    normDataSet = np.zeros(np.shape(dataSet))
    #返回dataSet的行数
    m = dataSet.shape[0]
    #用原始数据集减去最小值
    normDataSet = dataSet-np.tile(min_value,(m,1)) 
    #用normDataSet除以最大值减去最小的差，就得到归一化后的数据
    normDataSet = normDataSet/np.tile(value_range,(m,1))
    #返回归一化数据结果，数据范围，最小值
    return normDataSet,value_range,min_value

# =============================================================================
# if __name__ == '__main__':
#     #打开的文件名
#     filename = "datingTestSet.txt"
#     #打开并处理数据
#     datingDataMat, datingLabels = file2matrix(filename)
#     normDataSet, ranges, minVals = autoNorm(datingDataMat)
#     print(normDataSet)
#     print(ranges)
#     print(minVals)
# =============================================================================\
import numpy as np
import operator    
#对算法进行测试，在误差范围内，就使用算法，在测试前先编写出算法，10%的数据作为测试集，90%的数据作为数据集
def classify0(inX,dataSet,labels,k):
    '''
    KNN近邻算法函数，选择距离最近的前k个数据点的分类情况
    参数：inX：要输入的测试数据，dataSet：数据集
    labels：分类标签，k：前k个数据
    return sortedClassCount[0][0]
    '''
    #第一步先计算点之间的距离
    dataSetsize = dataSet.shape[0]  #计算数据集行数
    diffMat = np.tile(inX,(dataSetsize,1))-dataSet  #构造计算差矩阵
    #计算平方，然后行相加，最后开平方
    sqdiffMat = diffMat**2
    sqDistances = sqdiffMat.sum(axis=1)  #axis=1行相加
    Distances = sqDistances**0.5
    #返回距离中从小到大的索引值，方便在接下来的标签列表里对标签进行引用，而不是输出从小到大排序的值
    sortedDistances = Distances.argsort()
    #构建一个记录类别次数的字典
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistances[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1  
        #对字典的数值进行排序
        sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
        return sortedClassCount[0][0]
    

def datingClassTest():
    #打开的文件名
    filename = "datingTestSet.txt"
    #将返回的特征矩阵和分类向量分别存储到datingDataMat和datingLabels中
    datingDataMat, datingLabels = file2matrix(filename)
    #取所有数据的百分之十
    hoRatio = 0.10
    #数据归一化,返回归一化后的矩阵,数据范围,数据最小值
    normMat, ranges, minVals = autoNorm(datingDataMat)
    #获得normMat的行数
    m = normMat.shape[0]
    #百分之十的测试数据的个数
    numTestVecs = int(m * hoRatio)
    #分类错误计数
    errorCount = 0.0

    for i in range(numTestVecs):
        #前numTestVecs个数据作为测试集,后m-numTestVecs个数据作为训练集
        classifierResult = classify0(normMat[i,:], normMat[numTestVecs:m,:],
            datingLabels[numTestVecs:m], 4)
        print("分类结果:%d\t真实类别:%d" % (classifierResult, datingLabels[i]))
        if classifierResult != datingLabels[i]:
            errorCount += 1.0
    print("错误率:%f%%" %(errorCount/float(numTestVecs)*100))

# =============================================================================
# 
# if __name__ == '__main__':
#     datingClassTest()
# =============================================================================
def classifyPerson():
    #输出结果
    resultList = ['讨厌','有些喜欢','非常喜欢']
    #三维特征用户输入
    precentTats = float(input("玩视频游戏所耗时间百分比:"))
    ffMiles = float(input("每年获得的飞行常客里程数:"))
    iceCream = float(input("每周消费的冰激淋公升数:"))
    #打开的文件名
    filename = "datingTestSet.txt"
    #打开并处理数据
    datingDataMat, datingLabels = file2matrix(filename)
    #训练集归一化
    normMat, ranges, minVals = autoNorm(datingDataMat)
    #生成NumPy数组,测试集
    inArr = np.array([precentTats, ffMiles, iceCream])
    #测试集归一化
    norminArr = (inArr - minVals) / ranges
    #返回分类结果
    classifierResult = classify0(norminArr, normMat, datingLabels, 3)
    #打印结果
    print("你可能%s这个人" % (resultList[classifierResult-1]))

if __name__ == '__main__':
    classifyPerson()
    
    

    












