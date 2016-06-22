#!/bin/env python
#-*- coding: utf8 -*-

import sys
import os
import re
import pickle
from   optparse import OptionParser
import numpy as np
import tensorflow as tf

import util
import model

# --verbose
VERBOSE = 0

if __name__ == '__main__':
	parser = OptionParser()
	parser.add_option("--verbose", action="store_const", const=1, dest="verbose", help="verbose mode")
	parser.add_option("-m", "--model", dest="model_dir", help="dir path to load model", metavar="model_dir")
	(options, args) = parser.parse_args()
	if options.verbose == 1 : VERBOSE = 1
	model_dir = options.model_dir
	if model_dir == None :
		parser.print_help()
		exit(1)
	dic_path = model_dir + '/' + 'dic.pickle'
		
	# config
	n_steps = 30                    # time steps
	padd = '\t'                     # special padding chracter
	with open(dic_path, 'rb') as handle :
		char_dic = pickle.load(handle)    # load dic
	n_input = len(char_dic)         # input dimension, vocab size
	n_hidden = 8                    # hidden layer size
	n_classes = 2                   # output classes,  space or not
	vocab_size = n_input
	
	x = tf.placeholder(tf.float32, [None, n_steps, n_input])
	y_ = tf.placeholder(tf.int32, [None, n_steps])
	early_stop = tf.placeholder(tf.int32)

	# LSTM layer
	# 2 x n_hidden length (state & cell)
	istate = tf.placeholder(tf.float32, [None, 2*n_hidden])
	weights = {
		'hidden' : model.weight_variable([n_input, n_hidden]),
		'out' : model.weight_variable([n_hidden, n_classes])
	}
	biases = {
		'hidden' : model.bias_variable([n_hidden]),
		'out': model.bias_variable([n_classes])
	}

	y = model.RNN(x, istate, weights, biases, n_hidden, n_steps, n_input, early_stop)

	batch_size = 1
	logits = tf.reshape(tf.concat(1, y), [-1, n_classes])

	NUM_THREADS = 1
	config = tf.ConfigProto(intra_op_parallelism_threads=NUM_THREADS,
			inter_op_parallelism_threads=NUM_THREADS,
			log_device_placement=False)
	sess = tf.Session(config=config)
	init = tf.initialize_all_variables()
	sess.run(init)
	saver = tf.train.Saver() # save all variables
	checkpoint_dir = model_dir
	ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
	if ckpt and ckpt.model_checkpoint_path :
		saver.restore(sess, ckpt.model_checkpoint_path)
		sys.stderr.write("model restored from %s\n" %(ckpt.model_checkpoint_path))
	else :
		sys.stderr.write("no checkpoint found" + '\n')
		sys.exit(-1)
	
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
			batch_xs, batch_ys, next_pos, count = util.next_batch(sentence, pos, char_dic, vocab_size, n_steps, padd)
			
			print 'window : ' + sentence[pos:pos+n_steps]
			print 'count : ' + str(count)
			print 'next_pos : ' + str(next_pos)
			print batch_ys
			
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
					if tag_vector[pos+j] == util.CLASS_1 : # 1
						if tag == util.CLASS_0 : # 1 -> 0
							sys.stderr.write("1->0\n")
							tag_vector[pos+j] = tag
					else : # 0
						if tag == util.CLASS_1 : # 0 -> 1
							sys.stderr.write("0->1\n")
							tag_vector[pos+j] = tag
				j += 1
			pos = next_pos
		# generate output using tag_vector
		print 'out = ' + util.to_sentence(tag_vector, sentence)

		i += 1

	sess.close()
