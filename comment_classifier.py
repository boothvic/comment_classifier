#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# comment_classifier.py
#
# Vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
# Python source code - replace this with a description
# of the code and write the code below this text
#

import numpy as np
import tensorflow as tf
import random
import pickle
from collections import Counter

import nltk
from nltk.tokenize import word_tokenize
"""
'I'm super man'
tokenize:
['I', ''m', 'super', 'man']
"""
from nltk.stem import WordNetLemmatizer
"""
词形还原(lemmatizer)，即把一个任何形式的英语单词还原到一般形式，
与词根还原不同(stemmer)，后者是抽取一个单词的词根。
"""


pos_file = 'pos.txt'
neg_file = 'neg.txt'

# 创建词汇表
def create_lexicon(pos_file, neg_file):
    lex = []
    # 读取文件
    def process_file(txtfile):
        with open(txtfile, 'r') as f:
            lex = []
            lines = f.readlines()
            #print(lines)
            for line in lines:
                words = word_tokenize(line.lower())
                lex += words
            return lex

    lex += process_file(pos_file)
    lex + process_file(neg_file)
    #print(len(lex))
    lemmatizer = WordNetLemmatizer()
    lex = [lemmatizer.lemmatize(word) for word in lex] # 词形还原(cats -> cat)

    word_count = Counter(lex)
    #print(word_count)
    lex = []
    for word in word_count:
        if word_count[word] < 2000 and word_count[word] > 20:
            lex.append(word)
    return lex

#lex 里保存了文本中出现过的单词
lex = create_lexicon(pos_file, neg_file)


# 把每条评论转换为向量, 转换原理：
# 假设lex为['woman', 'great', 'feel', 'actually', 'looking', 'latest',
# 'seen','is'] 当然实际上要大的多
# 评论'i think this movie is great' 转换为 [0,1,0,0,0,0,0,1],
# 把评论中出现的字在lex中标记，出现过的标记为1，其余标记为0
def normalize_dataset(lex):
    dataset = []
    # lex:词汇表；review:评论；clf:评论对应的分类，
    # [0,1]代表负面评论 [1,0]代表正面评论
    def string_to_vector(lex, review, clf):
        words = word_tokenize(review.lower())
        lemmatizer = WordNetLemmatizer()
        words = [lemmatizer.lemmatize(word) for word in words]

        features = np.zeros(len(lex))
        for word in words:
            if word in lex:
                features[lex.index(word)] = 1
        return [features, clf]

    with open(pos_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            one_sample = string_to_vector(lex, line, [1,0])
            dataset.append(one_sample)
    with open(neg_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            one_sample = string_to_vector(lex, line, [0,1])
            dataset.append(one_sample)

    #print(len(dataset))
    return dataset


dataset = normalize_dataset(lex)
random.shuffle(dataset)
"""
#把整理好的数据保存到文件，方便使用。到此完成了数据的整理工作
with open('save.pickle', 'wb') as f:
	pickle.dump(dataset, f)
"""

# 取样本的10%作为测试数据
test_size = int(len(dataset) * 0.1)

dataset = np.array(dataset)

train_dataset = dataset[:-test_size]
test_dataset = dataset[-test_size:]

# Feed-Forword Neural Network
# 定义每个层有多少'神经元'
n_input_layer = len(lex) # 输入层

n_layer_1 = 1000 # hide layer
n_layer_2 = 1000

n_output_layer = 2 # 输出层


# 定义待训练的神经网络
def neural_network(data):
    layer_1_w_b = {'w_':tf.Variable(tf.random_normal([n_input_layer, n_layer_1])),
                   'b_':tf.Variable(tf.random_normal([n_layer_1]))}
    layer_2_w_b = {'w_':tf.Variable(tf.random_normal([n_layer_1, n_layer_2])),
                   'b_':tf.Variable(tf.random_normal([n_layer_2]))}
    #定义输出层'神经元'的权重和biases
    layer_output_w_b = {'w_':tf.Variable(tf.random_normal([n_layer_2,n_output_layer])),
                        'b_':tf.Variable(tf.random_normal([n_output_layer]))}

    # w*x+b
    layer_1 = tf.add(tf.matmul(data, layer_1_w_b['w_']), layer_1_w_b['b_'])
    layer_1 = tf.nn.relu(layer_1) # 激活函数
    layer_2 = tf.add(tf.matmul(layer_1, layer_2_w_b['w_']), layer_2_w_b['b_'])
    layer_2 = tf.nn.relu(layer_2) # 激活函数
    layer_output = tf.add(tf.matmul(layer_2, layer_output_w_b['w_']), layer_output_w_b['b_'])

    return layer_output


# 每次使用50条数据进行训练
batch_size = 50

X = tf.placeholder('float', [None, len(train_dataset[0][0])])
Y = tf.placeholder('float')
# 使用数据训练神经网络
def train_neural_network(X, Y):
    predict = neural_network(X)
    cost_func = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predict, labels=Y))
    optimizer = tf.train.AdamOptimizer().minimize(cost_func)  #learning rate default 0.001

    epochs = 12
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())

        random.shuffle(train_dataset)
        train_x = train_dataset[:, 0]
        train_y = train_dataset[:, 1]
        epoch_loss = 0
        for epoch in range(epochs):
            i = 0
            while i < len(train_x):
                start = i
                end = i + batch_size

                batch_x = train_x[start:end]
                batch_y = train_y[start:end]

                _, c = session.run([optimizer, cost_func], feed_dict=
                                   {X:list(batch_x), Y:list(batch_y)})
                epoch_loss += c
                i += batch_size

            print(epoch, ' : ', epoch_loss)

        test_x = test_dataset[:, 0]
        test_y = test_dataset[:, 1]
        correct = tf.equal(tf.argmax(predict, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('accuracy: ',accuracy.eval({X:list(test_x), Y:list(test_y)}))

train_neural_network(X, Y)

