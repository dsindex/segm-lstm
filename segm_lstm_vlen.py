#!/bin/env python
#-*- coding: utf8 -*-

import sys
import re
import tensorflow as tf
from tensorflow.models.rnn import rnn, rnn_cell
import numpy as np

def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

def build_dictionary(sentences, padd) :
	char_rdic = []
	for sentence in sentences :
		for c in sentence :
			if c not in char_rdic : char_rdic.append(c)
	if padd not in char_rdic : char_rdic.append(padd)
	char_dic = {w: i for i, w in enumerate(char_rdic)} # char to id
	return char_rdic, char_dic

def one_hot(i, size) :
	return [ 1 if j == i else 0 for j in xrange(size) ]

def get_xy_data(sentence, pos, n_steps, padd) :
	slen = len(sentence)
	x_data = []
	y_data = []
	next_pos = -1
	count = 0
	i = pos
	while i < slen :
		c = sentence[i]
		x_data.append(c)
		next_c = None
		if i+1 < slen : next_c = sentence[i+1]
		if next_c == ' ' : 
			y_data.append(1) # next is space
		else : 
			y_data.append(0) # next is not space
		count += 1
		i += 1
		if count == n_steps : break
	if count == n_steps :
		if i == slen : 
			# reached end
			next_pos = -1
		if i < slen :
			# move prev space + 1
			j = i-1
			while j > 0 :
				c = sentence[j]
				if c == ' ' : break
				j -= 1
			if j <= i - 1 : 
				next_pos = j+1
	else :
		# padding
		diff = n_steps - count
		x_data += [padd]*diff
		y_data += [0]*diff
		next_pos = -1

	return x_data, y_data, next_pos

def next_batch(sentences, begin, pos, char_dic, vocab_size, n_steps, padd) :
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
	sentence = sentences[begin]
	x_data, y_data, next_pos = get_xy_data(sentence, pos, n_steps, padd)
	x_data = [char_dic[c] for c in x_data]
	x_data = [one_hot(i, vocab_size) for i in x_data]
	batch_xs.append(x_data)
	batch_ys.append(y_data)
	batch_xs = np.array(batch_xs, dtype='f')
	batch_ys = np.array(batch_ys, dtype='int32')
	return batch_xs, batch_ys, next_pos

def test_next_batch(sentences, char_dic, vocab_size, n_steps, padd) :
	begin = 0
	num_sentences = len(sentences)
	while begin < num_sentences :
		pos = 0
		while pos != -1 :
			batch_xs, batch_ys, next_pos = next_batch(sentences, begin, pos, char_dic, vocab_size, n_steps, padd)
			print 'next_pos : ' + str(next_pos) + '\t' + sentences[begin][pos:pos+n_steps]
			print batch_ys
			pos = next_pos
		begin += 1

S1 = re.compile(u'''[\s]+''')
def snorm(string) :
	return S1.sub(' ', string.replace('\t',' ')).strip()

sentences = [u'이것을 띄어쓰기하면 어떻게 될까요.',
             u'아버지가 방에 들어 가신다.',
			 u'SK이노베이션, GS, S-Oil, 대림산업, 현대중공업 등 대규모 적자를 내던 기업들이 극한 구조조정을 통해 흑자로 전환하거나 적자폭을 축소한 것이 영업이익 개선을 이끈 것으로 풀이된다.']
num_sentences = len(sentences)
sentences = [snorm(sentence) for sentence in sentences]

# config
learning_rate = 0.01
training_iters = 500

n_steps = 20                    # time steps
min_size = 5                    # if size of sentence is bellow this, do not training/inference
padd = '\t'                     # special padding chracter

char_rdic, char_dic = build_dictionary(sentences, padd)
n_input = len(char_dic)         # input dimension, vocab size
n_hidden = 8                    # hidden layer size
n_classes = 2                   # output classes,  space or not
vocab_size = n_input
'''
test_next_batch(sentences, char_dic, vocab_size, n_steps, padd)
'''
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

def RNN(_X, _istate, _weights, _biases, n_steps):
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
	_X = tf.split(0, n_steps, _X) # n_steps splits each of which contains (?, n_hidden)
	'''
	ex)
	i  split0  split1  split2 .... split(n-1)
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
while seq < training_iters :
	begin = seq % num_sentences
	pos = 0
	while pos != -1 :
		batch_xs, batch_ys, next_pos = next_batch(sentences, begin, pos, char_dic, vocab_size, n_steps, padd)
		'''
		print 'next_pos : ' + str(next_pos) + '\t' + sentences[begin][pos:pos+n_steps]
		print batch_ys
		'''
		c_istate = np.zeros((batch_size, 2*n_hidden))
		feed={x: batch_xs, y_: batch_ys, istate: c_istate}
		sess.run(optimizer, feed_dict=feed)
		pos = next_pos
	if seq % 10 == 0 : 
		print 'seq : %s' % seq + ',' + 'cost : %s' % sess.run(cost, feed_dict=feed)
	seq += 1





# inference
test_sentences = [u'이것을띄어쓰기하면어떻게될까요.',
                  u'아버지가방에들어가신다.',
				  u'기업들이극한 구조조정을통해 흑자로전환하거나']
test_sentences = [snorm(sentence) for sentence in test_sentences]

batch_size = 1
i = 0
while i < len(test_sentences) :
	begin = i
	sentence = test_sentences[begin]
	sentence_size = len(sentence)
	tag_vector = [0]*(sentence_size+n_steps) # buffer n_steps
	pos = 0
	while pos != -1 :
		batch_xs, batch_ys, next_pos = next_batch(test_sentences, begin, pos, char_dic, vocab_size, n_steps, padd)
		'''
		print 'next_pos : ' + str(next_pos) + '\t' + sentence[pos:pos+n_steps]
		'''
		c_istate = np.zeros((batch_size, 2*n_hidden))
		feed={x: batch_xs, y_: batch_ys, istate: c_istate}
		result = sess.run(tf.arg_max(logits, 1), feed_dict=feed)
		# overlapped copy
		j = 0
		result_size = len(result)
		while j < result_size :
			tag = result[j]
			tag_vector[pos+j] = tag
			j += 1
		pos = next_pos
	# generate output using tag_vector(space or not)
	print 'out = ' + to_sentence(tag_vector, sentence)

	i += 1

def to_sentence(tag_vector, sentence) :
	out = []
	j = 0
	tag_vector_size = len(tag_vector)
	sentence_size = len(sentence)
	while j < tag_vector_size and j < sentence_size :
		tag = tag_vector[j]
		if tag == 1 :
			out.append(sentence[j])
			if sentence[j] != ' ' : out.append(' ')
		else :
			out.append(sentence[j])
		j += 1
	n_sentence = ''.join(out)
	return snorm(n_sentence.encode('utf-8'))


