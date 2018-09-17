# -*- coding: utf-8 -*-
'''
Created on 2018��9��13��

@author: WQ
'''

from com.zhanghao.util.seqlib import *
from com.zhanghao.util.data_process import *
from com.zhanghao.core.lstmNet import *
# reload(sys)
# sys.setdefaultencoding('utf-8')

train_word_num = pickle.load(open(WORKPATH + '/pfr/train_word_num.pickle', 'rb'))
train_label = pickle.load(open(WORKPATH + '/pfr/train_label.pickle', 'rb'))
nb_classes = len(np.unique(train_label))

init_weight_wv = pickle.load(open(WORKPATH + '/pfr/init_weight_wv.pickle', 'rb'))

#���������ʵ�
label_dict = dict(zip(np.unique(train_label), range(4)))
num_dict = {n:l for l, n in label_dict.items()}

#��Ŀ�����תΪ����
train_label = [label_dict[y] for y in train_label]
train_word_num = np.array(train_word_num)

#stacking LSTM
modelname = WORKPATH + '/pfr/my_model_weights.h5'
net = Lstm_Net()
net.init_weight = [np.array(init_weight_wv)]
net.nb_classes = nb_classes
net.splitset(train_word_num,train_label)
print('training...')
net.buildnet()
net.train(modelname)