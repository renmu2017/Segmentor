# -*- coding: utf-8 -*-
'''
Created on 2018��9��13��

@author: WQ
'''
from com.zhanghao.util.seqlib import *
from tensorflow.python.keras._impl.keras.layers.core import Dense, Activation
from tensorflow.python.keras._impl.keras.utils import np_utils
#����lstm������
class Lstm_Net(object):
    def __init__(self):
        self.init_weight = []
        self.batch_size = 128
        self.word_dim = 100
        self.maxlen = 7
        self.hidden_units = 100
        self.nb_classes = 0
    
    def buildnet(self):
        self.maxfeatures = self.init_weight[0].shape[0]
        self.model = Sequential()
        print('stacking LSTM......')
        self.model.add(Embedding(self.maxfeatures, self.word_dim, input_length = self.maxlen))
        self.model.add(LSTM(units = self.hidden_units, return_sequences = True))
        self.model.add(LSTM(units = self.hidden_units, return_sequences = False))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(self.nb_classes))
        self.model.add(Activation('softmax'))
        self.model.compile(loss = 'categorical_crossentropy', optimizer = 'adam')
    
    def train(self,modelname):
        result = self.model.fit(self.train_X, self.Y_train, batch_size = self.batch_size, \
                                nb_epoch = 2, validation_data = (self.test_X, self.Y_test))
        self.model.save_weights(modelname)
        
    def splitset(self,train_word_num, train_label, train_size =0.9, random_state = 1):
        self.train_X, self.test_X, train_y, test_y = train_test_split(train_word_num, train_label, train_size = 0.9, random_state = 1)
        self.Y_train = np_utils.to_categorical(train_y, self.nb_classes)
        self.Y_test = np_utils.to_categorical(test_y, self.nb_classes)
        
    def predict_num(self, input_num, input_txt, label_dict = '',num_dict = ''):
        input_num = np.array(input_num)
        predict_prob = self.model.predict_proba(input_num, verbose = False)
        predict_lable = self.model.predict_classes(input_num, verbose = False)
        for i, lable in enumerate(predict_lable[:-1]):
            if i == 0:
                predict_prob[i, label_dict[u'E']] = 0
                predict_prob[i, label_dict[u'M']] = 0
            if lable == label_dict[u'B']:
                predict_prob[i+1, label_dict[u'B']] = 0
                predict_prob[i+1, label_dict[u'S']] = 0
            if lable == label_dict[u'E']:
                predict_prob[i+1, label_dict[u'M']] = 0
                predict_prob[i+1, label_dict[u'E']] = 0
            if lable == label_dict[u'M']:
                predict_prob[i+1, label_dict[u'B']] = 0
                predict_prob[i+1, label_dict[u'S']] = 0
            if lable == label_dict[u'S']:
                predict_prob[i+1, label_dict[u'M']] = 0
                predict_prob[i+1, label_dict[u'E']] = 0
            predict_lable[i+1] = predict_prob[i+1].argmax()
        predict_lable_new = [num_dict[x] for x in predict_lable]
        result = [w+'/'+l for w, l in zip(input_txt, predict_lable_new)]
        return ' '.join(result) + '\n'
    
    def getweights(self, wfname):
        return self.model.load_weights(wfname)
    
    