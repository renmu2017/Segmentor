# -*- coding: utf-8 -*-
'''
Created on 2018年9月13日

@author: WQ
'''

from com.zhanghao.util.seqlib import *
from com.zhanghao.util.data_process import *
from com.zhanghao.core.lstmNet import *

# reload(sys)
# sys.setdefaultencoding('utf-8')

word2idx = pickle.load(open(WORKPATH + '/pfr/word2idx.pickle','rb'))
train_word_num = pickle.load(open(WORKPATH + '/pfr/train_word_num.pickle', 'rb'))
train_label = pickle.load(open(WORKPATH + '/pfr/train_label.pickle', 'rb'))
nb_classes = len(np.unique(train_label))

init_weight_wv = pickle.load(open(WORKPATH + '/pfr/init_weight_wv.pickle', 'rb'))

#建立两个字典
label_dict = dict(zip(np.unique(train_label), range(4)))
num_dict = {n:l for l,n in label_dict.items()}


# label_dict = {'S':0,'B':1,'M':2,'E':3}
# num_dict = {n:l for l, n in label_dict.items()}

temp_txt = u'校长说衣服上除了校徽别别别的'
temp_txt = list(temp_txt)
temp_num = featContext(temp_txt,word2idx = word2idx)

net = Lstm_Net()
net.init_weight = [np.array(init_weight_wv)]
net.nb_classes = nb_classes
net.buildnet()
net.getweights(WORKPATH + '/pfr/my_model_weights.h5')
temp = net.predict_num(temp_num, temp_txt, label_dict = label_dict, num_dict = num_dict)
print(normalPrint(temp))