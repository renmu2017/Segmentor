# -*- coding: utf-8 -*-
'''
Created on 2018��  9��12��

@author: WQ
'''

from com.zhanghao.util.seqlib import *
from asn1crypto.core import InstanceOf
from com.zhanghao.util.seqlib import *




def load_file(input_file):
    '''
    读取训练文件
    '''
    input_data = codecs.open(input_file,'r')
    input_text = input_data.read()
    return input_text

def trainW2V(corpus,epochs = 20, num_feature = 100, sg = 1, \
              min_word_count = 1, num_workers = 4, \
              context = 4, sample = 1e-5, negative = 5):
    w2v = word2vec.Word2Vec(workers = num_workers,sample = sample, \
                            size = num_feature, min_count = min_word_count, window = context)
    np.random.shuffle(corpus)
    w2v.build_vocab(corpus)
    for epoch in range(epochs):
        print('epoch'+str(epoch))
        np.random.shuffle(corpus)
        w2v.train(corpus,total_examples = len(corpus),epochs = epochs)
        w2v.alpha *= 0.9
        w2v.min_alpha = w2v.alpha
    print('word2vec DONE!')
    return w2v

def remove_tag(str):
    return str.split('/')[0]

def remove_list_tag(list):
    return [remove_tag(s) for s in list]

def check_w2v(w2v):
    print('--------------')
    print(w2v['国王']-w2v['王后'])
    print(w2v['男人']-w2v['女人'])
    print('---------------')

def freq_func(input_txt):
    corpus = nltk.Text(input_txt)
    fdist = nltk.probability.FreqDist(corpus)
    w = fdist.keys()
    v = fdist.values()
    
    freqdf = pd.DataFrame({'word': list(w), 'freq': list(v)})
    freqdf.sort_values(by='freq', ascending=False, inplace=True)
    freqdf['idx'] = np.arange(len(v))

    return freqdf
    
def initweightlist(w2v, idx2word, word2idx):
    init_weight_wv = []
    for i in range(len(idx2word)):
        init_weight_wv.append(w2v[idx2word[i]])
    #'U'为未登录字，'P'为两头padding用途，并增加两个相应的向量表示
    char_num = len(init_weight_wv)
    idx2word[char_num] = u'U'
    word2idx[u'U'] = char_num
    idx2word[char_num + 1] = u'P'
    word2idx[u'P'] = char_num + 1
    init_weight_wv.append(np.random.randn(100,))
    init_weight_wv.append(np.zeros(100,))
    return init_weight_wv, idx2word, word2idx

def character_tagging(input_file, output_file):
    input_data = codecs.open(input_file, 'r')
    output_data = codecs.open(output_file, 'w', 'utf-8')
    for line in input_data.readlines():
        word_list = line.strip().split()
        for word in word_list:
            if len(word) == 1:
                output_data.write(word+ "/S ")
            else:
                output_data.write(word[0] +"/B ")
                for w in word[1:len(word) - 1]:
                    output_data.write(w + "/M ")
                output_data.write(word[len(word) - 1] + "/E ")
        output_data.write("\n")
    input_data.close()
    output_data.close()
    
def featContext(sentence, word2idx = '', context = 7):
    predict_word_num = []
    for w in sentence:
        if w in word2idx:
            predict_word_num.append(word2idx[w])
        else:
            predict_word_num.append(word2idx[u'U'])
    num = len(predict_word_num)
    pad = int((context - 1) * 0.5)
    for i in range(pad):
        predict_word_num.insert(0, word2idx[u'P'])
        predict_word_num.append(word2idx[u'P'])
    train_x = []
    for i in range(num):
        train_x.append(predict_word_num[i:i+context])
    return train_x

def normalizecorpus():
    input = codecs.open('E:/WorkSpace0906/Segmentor/pfr/199801.txt','r')
    output = codecs.open('E:/WorkSpace0906/Segmentor/pfr/corpus.txt','w')
    for line in input.readlines():
        output.write(' '.join(remove_list_tag(line.strip().split()[1:])))
        output.write("\n")
    input.close()
    output.close()

def normalPrint(str):
    normalStr = []
    for word in str.split():
        temp = word.split('/')
        if temp[1] == 'B':
            normalStr.append(temp[0])
        elif temp[1] =='S':
            normalStr.append(' '+temp[0]+' ')
        elif temp[1] =='M':
            normalStr.append(temp[0])
        elif temp[1] == 'E':
            normalStr.append(temp[0]+' ')
    return ''.join(normalStr)
if __name__ == '__main__':
    print(normalPrint('南/B 京/E 市/B 长/E 江/S 大/B 桥/E 建/B 成/E 3/B 0/E 周/B 年/E 纪/B 念/E'))
    # normalizecorpus()
    # corpuspath = 'E:/WorkSpace0906/Segmentor/pfr/199801.txt'
    # input_text = load_file(corpuspath)
    #
    # # 计算词频
    # txtnltk = []
    # for w in input_text.split('\n'):
    #     txtnltk.extend(remove_tag(s) for s in w.split()[1:])
    # freqdf = freq_func(txtnltk)

