# -*- coding: utf-8 -*-
'''
Created on 2018��9��12��

@author: WQ
'''

from com.zhanghao.util.seqlib import *
from com.zhanghao.util.data_process import *

corpuspath = 'E:/WorkSpace0906/Segmentor/pfr/199801.txt'
input_text = load_file(corpuspath)

#word2vec 是一个二维数组
txtwv = [remove_list_tag(line.strip().split()[1:]) for line in input_text.split('\n') if line != '']
standard_corpus = 'E:/WorkSpace0906/Segmentor/pfr/corpus.txt'


w2v = trainW2V(txtwv)
w2v.save('E:/WorkSpace0906/Segmentor/pfr/wordvector.bin')
