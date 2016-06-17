#!/bin/env python
#-*- coding: utf8 -*-

import sys
import re
import pickle
from   optparse import OptionParser
import numpy as np
import tensorflow as tf
from   tensorflow.models.rnn import rnn, rnn_cell

# --verbose
VERBOSE = 0

CLASS_1 = 1  # next is space
CLASS_0 = 0  # next is not space

S1 = re.compile(u'''[\s]+''')

def open_file(filename, mode) :
	try : fid = open(filename, mode)
	except :
		sys.stderr.write("open_file(), file open error : %s\n" % (filename))
		exit(1)
	else :
		return fid

def close_file(fid) :
	fid.close()

def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

def build_dictionary(train_path, padd) :
	char_rdic = []
	visit = {}
	fid = open_file(train_path, 'r')
	for line in fid :
		line = line.strip()
		if line == "" : continue
		line = line.decode('utf-8')
		for c in line :
			if c not in visit : 
				char_rdic.append(c)
				visit[c] = 1
	if padd not in visit : char_rdic.append(padd)
	char_dic = {w: i for i, w in enumerate(char_rdic)} # char to id
	close_file(fid)
	return char_dic

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
			y_data.append(CLASS_1)
		else : 
			y_data.append(CLASS_0)
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
			space_count = 0
			while j > 0 :
				c = sentence[j]
				if c == ' ' :
					space_count += 1
					if space_count == 1 : break
				j -= 1
			if j <= i - 1 : 
				next_pos = j+1
	else :
		# padding
		diff = n_steps - count
		x_data += [padd]*diff
		y_data += [CLASS_0]*diff
		next_pos = -1
	return x_data, y_data, next_pos, count

def next_batch(sentence, pos, char_dic, vocab_size, n_steps, padd) :
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
	x_data, y_data, next_pos, count = get_xy_data(sentence, pos, n_steps, padd)
	x_data = [ char_dic[c] if c in char_dic else padd for c in x_data]
	x_data = [one_hot(i, vocab_size) for i in x_data]
	batch_xs.append(x_data)
	batch_ys.append(y_data)
	batch_xs = np.array(batch_xs, dtype='f')
	batch_ys = np.array(batch_ys, dtype='int32')
	return batch_xs, batch_ys, next_pos, count
	
def snorm(string) :
	return S1.sub(' ', string.replace('\t',' ')).strip()

def test_next_batch(train_path, char_dic, vocab_size, n_steps, padd) :
	fid = open_file(train_path, 'r')
	for line in fid :
		line = line.strip()
		if line == "" : continue
		line = line.decode('utf-8')
		sentence = snorm(line)
		pos = 0
		while pos != -1 :
			batch_xs, batch_ys, next_pos, count = next_batch(sentence, pos, char_dic, vocab_size, n_steps, padd)
			print 'window : ' + sentence[pos:pos+n_steps]
			print 'count : ' + str(count)
			print 'next_pos : ' + str(next_pos)
			print batch_ys
			pos = next_pos
	close_file(fid)

def RNN(_X, _istate, _weights, _biases, n_steps, early_stop):
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
	outputs, states = rnn.rnn(lstm_cell, _X, initial_state=_istate, sequence_length=early_stop)
	final_outputs = []
	for output in outputs :
		# Linear activation
		final_output = tf.matmul(output, _weights['out']) + _biases['out'] # (?, n_classes)
		final_outputs.append(final_output)
	return final_outputs

def to_sentence(tag_vector, sentence) :
	out = []
	j = 0
	tag_vector_size = len(tag_vector)
	sentence_size = len(sentence)
	while j < tag_vector_size and j < sentence_size :
		tag = tag_vector[j]
		if tag == CLASS_1 :
			out.append(sentence[j])
			if sentence[j] != ' ' : out.append(' ')
		else :
			out.append(sentence[j])
		j += 1
	n_sentence = ''.join(out)
	return snorm(n_sentence).encode('utf-8')


if __name__ == '__main__':
	parser = OptionParser()
	parser.add_option("--verbose", action="store_const", const=1, dest="verbose", help="verbose mode")
	parser.add_option("-t", "--train", dest="train_path", help="train file path", metavar="train_path")
	parser.add_option("-d", "--dic", dest="dic_path", help="dic file path(will be saved)", metavar="dic_path")
	(options, args) = parser.parse_args()
	if options.verbose == 1 : VERBOSE = 1
	train_path = options.train_path
	if train_path == None :
		parser.print_help()
		exit(1)
	dic_path = options.dic_path
	if dic_path == None :
		parser.print_help()
		exit(1)

	# config
	n_steps = 20                    # time steps
	padd = '\t'                     # special padding chracter
	char_dic = build_dictionary(train_path, padd)
	n_input = len(char_dic)         # input dimension, vocab size
	n_hidden = 8                    # hidden layer size
	n_classes = 2                   # output classes,  space or not
	vocab_size = n_input

	test_next_batch(train_path, char_dic, vocab_size, n_steps, padd)

	x = tf.placeholder(tf.float32, [None, n_steps, n_input])
	y_ = tf.placeholder(tf.int32, [None, n_steps])
	early_stop = tf.placeholder(tf.int32)

	# LSTM layer
	# 2 x n_hidden length (state & cell)
	istate = tf.placeholder(tf.float32, [None, 2*n_hidden])
	weights = {
		'hidden' : weight_variable([n_input, n_hidden]),
		'out' : weight_variable([n_hidden, n_classes])
	}
	biases = {
		'hidden' : bias_variable([n_hidden]),
		'out': bias_variable([n_classes])
	}

	# training
	y = RNN(x, istate, weights, biases, n_steps, early_stop)

	batch_size = 1
	learning_rate = 0.01
	training_iters = 50
	logits = tf.reshape(tf.concat(1, y), [-1, n_classes])
	targets = y_
	seq_weights = tf.ones([n_steps * batch_size])
	loss = tf.nn.seq2seq.sequence_loss_by_example([logits], [targets], [seq_weights])
	cost = tf.reduce_sum(loss) / batch_size 
	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

	NUM_THREADS = 1
	config = tf.ConfigProto(intra_op_parallelism_threads=NUM_THREADS,
			inter_op_parallelism_threads=NUM_THREADS,
			log_device_placement=False)
	sess = tf.Session(config=config)
	init = tf.initialize_all_variables()
	sess.run(init)

	i = 0
	fid = open_file(train_path, 'r')
	for line in fid :
		line = line.strip()
		if line == "" : continue
		line = line.decode('utf-8')
		sentence = snorm(line)
		seq = 0
		while seq < training_iters :
			pos = 0
			while pos != -1 :
				batch_xs, batch_ys, next_pos, count = next_batch(sentence, pos, char_dic, vocab_size, n_steps, padd)
				'''
				print 'window : ' + sentences[begin][pos:pos+n_steps]
				print 'count : ' + str(count)
				print 'next_pos : ' + str(next_pos)
				print batch_ys
				'''
				c_istate = np.zeros((batch_size, 2*n_hidden))
				feed={x: batch_xs, y_: batch_ys, istate: c_istate, early_stop:count}
				sess.run(optimizer, feed_dict=feed)
				pos = next_pos
			if seq % 10 == 0 : 
				print '(i,seq) : (%s,%s)' % (i,seq) + ',' + 'cost : %s' % sess.run(cost, feed_dict=feed)
			seq += 1
		i += 1
	close_file(fid)
	# save dic
	with open(dic_path, 'wb') as handle :
		pickle.dump(char_dic, handle)

	print 'end of training'
	# ------------------------------------------------------------------------------------

	# inference

	# config
	n_steps = 20                    # time steps
	padd = '\t'                     # special padding chracter
	with open(dic_path, 'rb') as handle :
		char_dic = pickle.load(handle)    # load dic
	n_input = len(char_dic)         # input dimension, vocab size
	n_hidden = 8                    # hidden layer size
	n_classes = 2                   # output classes,  space or not
	vocab_size = n_input
	'''
	x = tf.placeholder(tf.float32, [None, n_steps, n_input])
	y_ = tf.placeholder(tf.int32, [None, n_steps])
	early_stop = tf.placeholder(tf.int32)

	# LSTM layer
	# 2 x n_hidden length (state & cell)
	istate = tf.placeholder(tf.float32, [None, 2*n_hidden])
	weights = {
		'hidden' : weight_variable([n_input, n_hidden]),
		'out' : weight_variable([n_hidden, n_classes])
	}
	biases = {
		'hidden' : bias_variable([n_hidden]),
		'out': bias_variable([n_classes])
	}

	y = RNN(x, istate, weights, biases, n_steps, early_stop)

	batch_size = 1
	logits = tf.reshape(tf.concat(1, y), [-1, n_classes])
	'''
	batch_size = 1
	i = 0
	while 1 :
		try : line = sys.stdin.readline()
		except KeyboardInterrupt : break
		if not line : break
		line = line.strip()
		if not line : continue
		line = line.decode('utf-8')
		sentence = line
		sentence_size = len(sentence)
		tag_vector = [-1]*(sentence_size+n_steps) # buffer n_steps
		pos = 0
		while pos != -1 :
			batch_xs, batch_ys, next_pos, count = next_batch(sentence, pos, char_dic, vocab_size, n_steps, padd)
			'''	
			print 'window : ' + sentence[pos:pos+n_steps]
			print 'count : ' + str(count)
			print 'next_pos : ' + str(next_pos)
			print batch_ys
			'''
			c_istate = np.zeros((batch_size, 2*n_hidden))
			feed={x: batch_xs, y_: batch_ys, istate: c_istate, early_stop:count}
			result = sess.run(tf.arg_max(logits, 1), feed_dict=feed)
			# overlapped copy and merge
			j = 0
			result_size = len(result)
			while j < result_size :
				tag = result[j]
				if tag_vector[pos+j] == -1 :
					tag_vector[pos+j] = tag
				else :
					if tag_vector[pos+j] == CLASS_1 : # 1
						if tag == CLASS_0 : # 1 -> 0
							sys.stderr.write("1->0\n")
							tag_vector[pos+j] = tag
					else : # 0
						if tag == CLASS_1 : # 0 -> 1
							sys.stderr.write("0->1\n")
							tag_vector[pos+j] = tag
				j += 1
			pos = next_pos
		# generate output using tag_vector
		print 'out = ' + to_sentence(tag_vector, sentence)

		i += 1



