#!/usr/bin/ python
# -*- coding: utf-8 -*-
# @Time    : 2018/10/11 17:26
# @Author  : Nanzhi.Wang
# @User    : wnz
# @Site    : https://github.com/rango94
# @File    : functions.py
# @Software: PyCharm
import numpy as np

def lookup(pre,std,num=30):
    for i in range(num):
        print(pre[i],'|',std[i])


def cont_pre(score,std):
    sum=0
    for idx in range(len(score)):
        if np.argmax(score[idx])==std[idx]:
            sum+=1
    return sum/len(score)