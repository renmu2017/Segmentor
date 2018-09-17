# -*- coding: utf-8 -*-
'''
Created on 2018��9��13��

@author: WQ
'''
from com.zhanghao.util.seqlib import *
from com.zhanghao.util.data_process import *
# reload(sys)
# sys.setdefaultencoding('utf-8')

corpuspath = 'E:/WorkSpace0906/Segmentor/pfr/199801.txt'
standardcorpus = 'E:/WorkSpace0906/Segmentor/pfr/corpus.txt'
input_text = load_file(corpuspath)

#计算词频
txtnltk = []
for w in input_text.split('\n'):
    txtnltk.extend(remove_tag(s) for s in w.split()[1:])
freqdf = freq_func(txtnltk)

#建立两个映射词典
word2idx = dict((c,i) for c, i in zip(freqdf.word,freqdf.idx))
idx2word = dict((i,c) for c, i in zip(freqdf.word,freqdf.idx))
w2v = word2vec.Word2Vec.load('E:/WorkSpace0906/Segmentor/pfr/wordvector.bin')

#初始化向量
init_weight_wv, idx2word, word2idx = initweightlist(w2v, idx2word, word2idx)
pickle.dump(word2idx, open('E:/WorkSpace0906/Segmentor/pfr/word2idx.pickle', 'wb'))
pickle.dump(idx2word, open('E:/WorkSpace0906/Segmentor/pfr/idx2word.pickle', 'wb'))

pickle.dump(init_weight_wv, open('E:/WorkSpace0906/Segmentor/pfr/init_weight_wv.pickle', 'wb'))

#读取数据，将格式进行转换为带4种标签S B M E
output_file = 'E:/WorkSpace0906/Segmentor/pfr/pfr.tagging.utf8'
character_tagging(standardcorpus, output_file)

#分离word 和label
with open(output_file, encoding= 'utf-8') as f:
    lines =f.readlines()
    cpprint(lines[10:15])
    train_line = [[w[0] for w in line.split()] for line in lines]
    train_label = [w[2] for line in lines for w in line.split()]
    print('train_label')
    cpprint(train_label[:100])
    print('train_line')
    cpprint(train_line[:100])

#将所有训练文本转成数字list
train_word_num = []
for line in train_line:
    train_word_num.extend(featContext(line, word2idx))

#持久化
pickle.dump(train_word_num, open('E:/WorkSpace0906/Segmentor/pfr/train_word_num.pickle', 'wb'))
pickle.dump(train_label, open('E:/WorkSpace0906/Segmentor/pfr/train_label.pickle', 'wb'))
