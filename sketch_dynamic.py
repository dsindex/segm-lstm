#!/bin/env python
#-*- coding: utf8 -*-

import tensorflow as tf
import numpy as np
import sys

CLASS_1 = 1  # next is space
CLASS_0 = 0  # next is not space

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def build_dictionary(sentences) :
    idx2char = []
    for sentence in sentences :
        for c in sentence :
            if c not in idx2char : idx2char.append(c)
    char2idx = {w: i for i, w in enumerate(idx2char)} # char to id
    return idx2char, char2idx

def one_hot(i, size) :
    return [ 1 if j == i else 0 for j in xrange(size) ]

def next_batch(sentences, begin, batch_size, sequence_length, char2idx) :
    '''
    y_data =  1 or 0     => sequence_length unfolding => [0,0,1,0,...]
    ^
    |
    x_data = [1,0,...,0] => sequence_length unfolding => [[1,0,..0],..,[0,0,1,..0]]

    batch_xs.shape => (batch_size, sequence_length, input_dim)
    batch_ys.shape => (batch_size, sequence_length)
    '''
    batch_xs = []
    batch_ys = []
    count = 0
    vocab_size = len(char2idx)
    for sentence in sentences[begin:] :
        x_data = sentence[0:sequence_length]
        x_data = [char2idx[c] for c in x_data]
        x_data = [one_hot(i, vocab_size) for i in x_data]
        batch_xs.append(x_data)
        y_data = []
        for c in sentence[1:sequence_length] :
            if c == ' ' : y_data.append(CLASS_1) # next is space
            else : y_data.append(CLASS_0)        # next is not space
        y_data.append(CLASS_0)
        batch_ys.append(y_data)
        count += 1
        if count == batch_size : break
    batch_xs = np.array(batch_xs, dtype='f')
    batch_ys = np.array(batch_ys, dtype='int32')
    return batch_xs, batch_ys

def rnn_model(hidden_sizie, batch_size, X) :
    cell = tf.contrib.rnn.LSTMCell(num_units=hidden_size, state_is_tuple=True)
    outputs, _= tf.nn.dynamic_rnn(cell=cell, inputs=X, dtype=tf.float32)
    return outputs

sentences = [u'이것을 띄어쓰기하면 어떻게 될까요.',
             u'아버지가 방에 들어가신다.']
sequence_length = len(sentences[0])     # time stpes
# padding
i = 0
while i < len(sentences) :
    sentence = sentences[i]
    length = len(sentence)
    diff = sequence_length - length
    if diff > 0 : # add padding
        sentences[i] += ' '*diff
    i += 1

# config
learning_rate = 0.01
training_iters = 10000

idx2char, char2idx = build_dictionary(sentences)
vocab_size = len(char2idx)
input_dim = vocab_size          # input dimension, one-hot size, vocab size
n_classes = 2                   # output classes,  space or not
hidden_size = n_classes         # output form LSTM, directly predict one-hot

X = tf.placeholder(tf.float32, [None, sequence_length, input_dim])  # X one-hot, (None, 19, 25)
Y = tf.placeholder(tf.int32, [None, sequence_length])               # Y label,   (None, 19)

# training
batch_size = 1
outputs = rnn_model(hidden_size, batch_size, X)  # (None, 19, 2)
weights = tf.ones([batch_size, sequence_length])
sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=outputs, targets=Y, weights=weights)
loss = tf.reduce_mean(sequence_loss)
train = tf.train.AdamOptimizer(learning_rate=0.1).minimize(loss)
prediction = tf.argmax(outputs, axis=2)          # (None, 19)

NUM_THREADS = 1
sess = tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=NUM_THREADS,inter_op_parallelism_threads=NUM_THREADS,log_device_placement=False))
init = tf.global_variables_initializer()
sess.run(init)

step = 0
while step < training_iters :
    begin = (step % (len(sentences)/batch_size)) * batch_size
    batch_xs, batch_ys = next_batch(sentences, begin, batch_size, sequence_length, char2idx)
    '''
    print 'batch_xs.shape : ' + str(batch_xs.shape)
    print 'batch_xs : '
    print batch_xs
    print 'batch_ys.shape : ' + str(batch_ys.shape)
    print 'batch_ys : '
    print batch_ys
    '''
    l, _ = sess.run([loss, train], feed_dict={X: batch_xs, Y: batch_ys})
    results = sess.run(prediction, feed_dict={X: batch_xs})
    if step % 50 == 0 :
        print(step, "loss:", l, "prediction: ", results, "true Y: ", batch_ys)
    step += 1

# inference
test_sentences = [u'이것을띄어쓰기하면어떻게될까요.',
                  u'아버지가방에들어가신다.']
# padding
i = 0
while i < len(test_sentences) :
    sentence = test_sentences[i]
    length = len(sentence)
    diff = sequence_length - length
    if diff > 0 : # add padding
        test_sentences[i] += ' '*diff
    i += 1
    
batch_size = len(test_sentences)
begin = 0
batch_xs, batch_ys = next_batch(test_sentences, begin, batch_size, sequence_length, char2idx)

feed_dict={X: batch_xs, Y: batch_ys}
results = sess.run(prediction, feed_dict={X: batch_xs})

i = 0
while i < len(test_sentences) :
    sentence = test_sentences[i]
    bidx = 0
    eidx = sequence_length
    rst = results[i][bidx:eidx]
    print 'rst = ', rst
    # generate output using tag(space or not)
    out = []
    j = 0
    while j < sequence_length :
        print 'rst[j] = ', rst[j]
        tag = rst[j]
        if tag == CLASS_1 :
            out.append(sentence[j])
            out.append(' ')
        else :
            out.append(sentence[j])
        j += 1
    n_sentence = ''.join(out).strip()
    print 'out = ' + n_sentence
    i += 1

