#!/bin/env python
#-*- coding: utf8 -*-

import tensorflow as tf
from tensorflow.models.rnn import rnn, rnn_cell
import numpy as np

def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

def build_dictionary(sentences) :
	char_rdic = []
	for sentence in sentences :
		for c in sentence :
			if c not in char_rdic : char_rdic.append(c)
	char_dic = {w: i for i, w in enumerate(char_rdic)} # char to id
	return char_rdic, char_dic

def one_hot(i, size) :
	return [ 1 if j == i else 0 for j in xrange(size) ]

def next_batch(sentences, begin, char_dic) :
	'''
	y_data =  1 or 0     => n_steps unfolding => [0,0,1,0,...]
	^
	|
	x_data = [1,0,...,0] => n_steps unfolding => [[1,0,..0],..,[0,0,1,..0]]

	batch_xs.shape => (batch_size=1, n_steps, n_input)
	batch_ys.shape => (batch_size=1, n_steps)
	'''
	batch_xs = []
	batch_ys = []
	vocab_size = len(char_dic)
	sentence = sentences[begin]
	n_steps = len(sentence)
	x_data = sentence
	x_data = [char_dic[c] for c in x_data]
	x_data = [one_hot(i, vocab_size) for i in x_data]
	batch_xs.append(x_data)
	y_data = []
	for c in sentence[1:] :
		if c == ' ' : y_data.append(1) # next is space
		else : y_data.append(0)        # next is not space
	y_data.append(0)
	batch_ys.append(y_data)
	batch_xs = np.array(batch_xs, dtype='f')
	batch_ys = np.array(batch_ys, dtype='int32')
	return batch_xs, batch_ys, n_steps


sentences = [u'이것을 띄어쓰기하면 어떻게 될까요.',
             u'아버지가 방에 들어가신다.']

# config
learning_rate = 0.01
training_iters = 1000

char_rdic, char_dic = build_dictionary(sentences)
n_input = len(char_dic)         # input dimension, vocab size
n_hidden = 8                    # hidden layer size
n_classes = 2                   # output classes,  space or not

n_steps = tf.placeholder("int32")
x = tf.placeholder("float", [None, n_steps, n_input])
y_ = tf.placeholder("int32", [None, n_steps])

# LSTM layer
# 2 x n_hidden length (state & cell)
istate = tf.placeholder("float", [None, 2*n_hidden])
weights = {
	'hidden' : weight_variable([n_input, n_hidden]),
	'out' : weight_variable([n_hidden, n_classes])
}
biases = {
	'hidden' : bias_variable([n_hidden]),
	'out': bias_variable([n_classes])
}

def RNN(_X, _istate, _weights, _biases, _n_steps):
	# input _X shape: (batch_size, n_steps, n_input)
	# switch n_steps and batch_size, (n_steps, batch_size, n_input)
	_X = tf.transpose(_X, [1, 0, 2])
	# Reshape to prepare input to hidden activation
	# (n_steps*batch_size, n_input) => (?, n_input)
	_X = tf.reshape(_X, [-1, n_input])
	# Linear activation
	_X = tf.matmul(_X, _weights['hidden']) + _biases['hidden'] # (?, n_hidden)

	# Define a lstm cell with tensorflow
	lstm_cell = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)
	# Split data because rnn cell needs a list of inputs for the RNN inner loop
	_X = tf.split(0, _n_steps, _X) # n_steps splits each of which contains (?, n_hidden)
	'''
	ex)
	i  split0  split1  split2 .... split(n)
	0  (8)     ...                 (8)
	1  (8)     ...                 (8)
	...
	m  (8)     ...                 (8)
	'''
	# Get lstm cell output
	outputs, states = rnn.rnn(lstm_cell, _X, initial_state=_istate)
	final_outputs = []
	for output in outputs :
		# Linear activation
		final_output = tf.matmul(output, _weights['out']) + _biases['out'] # (?, n_classes)
		final_outputs.append(final_output)
	return final_outputs


# training
y = RNN(x, istate, weights, biases, n_steps)

batch_size = 1
logits = tf.reshape(tf.concat(1, y), [-1, n_classes])
targets = y_
seq_weights = tf.ones([n_steps * batch_size])
loss = tf.nn.seq2seq.sequence_loss_by_example([logits], [targets], [seq_weights])
cost = tf.reduce_sum(loss) / batch_size 
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

NUM_THREADS = 1
sess = tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=NUM_THREADS,inter_op_parallelism_threads=NUM_THREADS,log_device_placement=False))
init = tf.initialize_all_variables()
sess.run(init)

seq = 0
while step < training_iters :
	begin = seq
	batch_xs, batch_ys, steps = next_batch(sentences, begin, char_dic)
	'''
	print 'batch_xs.shape : ' + str(batch_xs.shape)
	print 'batch_xs : '
	print batch_xs
	print 'batch_ys.shape : ' + str(batch_ys.shape)
	print 'batch_ys : '
	print batch_ys
	'''
	c_istate = np.zeros((batch_size, 2*n_hidden))
	feed={n_steps:steps, x: batch_xs, y_: batch_ys, istate: c_istate}
	sess.run(optimizer, feed_dict=feed)
	if seq % 10 == 0 : 
		print 'seq : %s' % seq + ',' + 'cost : %s' % sess.run(cost, feed_dict=feed)
	seq += 1

# inference
test_sentences = [u'이것을띄어쓰기하면어떻게될까요.',
                  u'아버지가방에들어가신다.']
	
i = 0
while i < len(test_sentences) :
	batch_size = 1
	begin = i
	batch_xs, batch_ys, steps = next_batch(test_sentences, begin, char_dic)
	c_istate = np.zeros((batch_size, 2*n_hidden))
	feed={n_steps:steps, x: batch_xs, y_: batch_ys, istate: c_istate}
	result = sess.run(tf.arg_max(logits, 1), feed_dict=feed)

	sentence = test_sentences[i]
	# generate output using tag(space or not)
	out = []
	j = 0
	while j < steps :
		tag = result[j]
		if tag == 1 :
			out.append(sentence[j])
			out.append(' ')
		else :
			out.append(sentence[j])
		j += 1
	n_sentence = ''.join(out).strip()
	print 'out = ' + n_sentence
	i += 1

