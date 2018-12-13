# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 14:15:14 2018

@author: user
"""
# =============================================================================
# 朴素贝叶斯算法一般流程：
# 1.准备数据集：已经准备好的txt文档
# 2.预处理数据集：将准备好的txt文件做成词条向量
# 3.测试数据集：查看词条向量是否符合标准
# 4.训练算法：构建训练模型
# 5.测试算法：测试算法的错误率
# 6.使用算法
# =============================================================================
import re
import numpy as np
import random

def textParse(bigString):
    '''
    把输入的每个文本文档（大的字符串）解析成字符串列表
    输入参数：每个文本文档（大的字符串）
    '''
    listOfTokens = re.split(r'\W*', bigString)  #将特殊字符作为切分标志对字符串进行切分
    return [tok.lower() for tok in listOfTokens if len(tok)>2]  #将除了单个字符以外的单词都改成小写


def createVocabList(dataSet):
    '''
    将切分的词条整理成不重复的词条列表，也就是词汇表
    输入参数：每个文本文档（大的字符串）
    '''
    vocabSet = set([])  #创建一个空的不重复列表
    for document in dataSet:
        vocabSet = vocabSet | set(document)  #取并集，得到完整的词汇表
    return list(vocabSet)

    
def setOfWords2Vec(vocablist,inputSet):
    '''
    根据已经整理好的词汇表，将每个文本文档里的词汇向量化，如果存在于词汇表里的单词就置为1，不存在则置为0
    输入参数：vacabList（createVocabList返回的列表）  inputSet（每个文本文档）
    输出参数：文档向量，词集模型
    '''
    returnVec = [0]*len(vocablist)  #创建一个所含元素都为0的向量
    for word in inputSet:  #遍历每个词条
        if word in vocablist:
            returnVec[vocablist.index(word)] = 1  #如果该词汇在词汇表中，那么在相应的位置标记为1
        else:
            print('the word :%s is not in the vocabulary' %word)
    return returnVec  #返回文档向量

def bagOfWords2VecMN(vocabList, inputSet):
    '''
    根据已经整理好的词汇表，将每个文本文档里的词汇向量化，如果存在于词汇表里的单词累计加一
    输入参数：vacabList（createVocabList返回的列表）  inputSet（每个文本文档）
    输出参数：文档向量，词集模型
    '''
    returnVec = [0]*len(vocabList)                                        #创建一个其中所含元素都为0的向量
    for word in inputSet:                                                #遍历每个词条
        if word in vocabList:                                            #如果词条存在于词汇表中，则计数加一
            returnVec[vocabList.index(word)] += 1
    return returnVec                                                    #返回词袋模型

def trainNB0(trainMatrix,trainCategory):
    '''
    朴素贝叶斯分类器训练函数
    输入参数：trainMatrix：setOfWords2Vec返回的文档向量  trainCategory：分类标签向量
    输出参数：p0Vect:非垃圾邮箱的概率数组
             p1Vect:垃圾邮箱的概率数组
             pAbusive:输入垃圾邮箱的概率
    '''
    numTrainDocs = len(trainMatrix)  #计算文档的数目
    numWords = len(trainMatrix[0])  #计算每个文档的单词数量
    pAbusive = sum(trainCategory) /float(numTrainDocs)  #计算垃圾邮箱的概率
    p0Num = np.ones(numWords)  #词条出现数初始化为1，拉普拉斯平滑
    p1Num = np.ones(numWords)  #词条出现数初始化为1，拉普拉斯平滑
    p0Denom = 2.0  #分母初始化为2，拉普拉斯平滑
    p1Denom = 2.0  #分母初始化为2，拉普拉斯平滑
    for i in range(numTrainDocs):  #遍历每个文档
        if trainCategory[i] == 1:  #统计垃圾邮箱条件概率所需要的数据
            p1Num += trainMatrix[i]  
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]  
            p0Denom += sum(trainMatrix[i])
    p1Vect = np.log(p1Num/p1Denom)
    p0Vect = np.log(p0Num/p1Denom)
    return pAbusive,p1Vect,p0Vect


def classifyNB(vec2Classify,p1Vect,p0Vect,pClass1):
    '''
    朴素贝叶斯分类器函数
    输入参数：vec2Classify：待分类的词条,p1Vect：垃圾邮箱的条件概率数组,
    p0Vect：非垃圾邮箱的条件概率数组,pClass1：垃圾邮箱的概率
    输出参数：1,0
    '''
    p1 = sum(p1Vect*vec2Classify) + np.log(pClass1)  #logA*B = logA + logB
    p0 = sum(p0Vect*vec2Classify) + np.log(1-pClass1)
    if p1 > p0:
        return 1
    else:
        return 0
    

def spamTest():
    '''
    测试朴素贝叶斯分类器函数
    '''
    docList = []    #词条列表
    class_List = []  #分类标签列表
    for i in range(1,26):  #遍历25个txt文件
        wordList = textParse(open('email/spam/%d.txt' %i,'r').read())  #读取每个文件，并且把它们转化为词条列表
        docList.append(wordList)
        class_List.append(1)
         
        wordList = textParse(open('email/ham/%d.txt' %i,'r').read())  #读取每个文件，并且把它们转化为词条列表
        docList.append(wordList)
        class_List.append(0)
    vocabList = createVocabList(docList)  #创建不重复词汇的词汇列表
    trainingSet = list(range(50))  #创建训练集的索引值的列表
    testSet = []  #测试集的索引值的列表
    for i in range(10):
        randIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat = []  #创建训练集矩阵
    trainClasses = []  #创建训练标签向量
    for docIndex in trainingSet:  #遍历训练集
        trainMat.append(setOfWords2Vec(vocabList,docList(docIndex)))  #将生成的词集添加到训练矩阵
        trainClasses.append(class_List(docIndex))  #将标签添加到标签向量
    pAbusive,p1Vect,p0Vect = trainNB0(np.array(trainMat),np.array(trainClasses))  #训练朴素贝叶斯模型
    errorCount = 0
    for docIndex in testSet:
        wordVector = setOfWords2Vec(vocabList,docList(docIndex))
        if classifyNB(np.array(wordVector),p1Vect,p0Vect,pAbusive) != trainClasses[i]:  #如果分类器的结果不等于列表里的结果
            errorCount += 1
            print('分类错误的测试集：',docList[docIndex])
    print('错误率：%.2f%%'%(float(errorCount)/len(testSet)*100))
    

if __name__ == '__main__':
    spamTest()
    
     
     



    
            
    


    
    
    
    
    
    
    
    
    
    
    
        











    