# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 10:06:52 2018

@author: user
"""

import numpy as np
import operator

def createDataSet():
    '''创建数据集以及对应的标签'''
    group = np.array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group,labels

