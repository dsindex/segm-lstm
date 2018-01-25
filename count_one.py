#Source code with the blog post at http://monik.in/a-noobs-guide-to-implementing-rnn-lstm-using-tensorflow/
import numpy as np
import random
from random import shuffle
import tensorflow as tf

# data preparation
NUM_EXAMPLES = 1000
train_input = ['{0:020b}'.format(i) for i in range(2**10)]
shuffle(train_input)
train_input = [map(int,i) for i in train_input]
ti  = []
for i in train_input:
    temp_list = []
    for j in i:
            temp_list.append([j])
    ti.append(np.array(temp_list))
train_input = ti

train_output = []
for i in train_input:
    count = 0
    for j in i:
        if j[0] == 1:
            count+=1
    temp_list = ([0]*21)
    temp_list[count]=1
    train_output.append(temp_list)

test_input = train_input[NUM_EXAMPLES:]
test_output = train_output[NUM_EXAMPLES:]
train_input = train_input[:NUM_EXAMPLES]
train_output = train_output[:NUM_EXAMPLES]

print "test and training data loaded"

# build model
data = tf.placeholder(tf.float32, [None, 20,1]) #Number of examples, number of input, dimension of each input
target = tf.placeholder(tf.float32, [None, 21])
num_hidden = 24
cell = tf.contrib.rnn.LSTMCell(num_hidden,state_is_tuple=True)
val, _ = tf.nn.dynamic_rnn(cell, data, dtype=tf.float32)
## (?, 20, 24)
print 'before transpose : ', val.get_shape()
val = tf.transpose(val, [1, 0, 2])
## (20, ?, 24)
print 'after transpose : ', val.get_shape()
## 20
print 'first dimension of val(number of input = sequence length) : ', int(val.get_shape()[0])
last = tf.gather(val, int(val.get_shape()[0]) - 1)
## (?, 24)
print 'last(dimension of each hidden) : ', last.get_shape()
weight = tf.Variable(tf.truncated_normal([num_hidden, int(target.get_shape()[1])]))
bias = tf.Variable(tf.constant(0.1, shape=[target.get_shape()[1]]))
output = tf.matmul(last, weight) + bias
## (?, 24) x (24, 20) -> (?, 20)
print 'output : ', output.get_shape()
prediction = tf.nn.softmax(output)
## (?, 20)
print 'prediction : ', prediction.get_shape()
## KL distance
cross_entropy = -tf.reduce_sum(target * tf.log(tf.clip_by_value(prediction,1e-10,1.0)))
optimizer = tf.train.AdamOptimizer()
minimize = optimizer.minimize(cross_entropy)
mistakes = tf.not_equal(tf.argmax(target, 1), tf.argmax(prediction, 1))
error = tf.reduce_mean(tf.cast(mistakes, tf.float32))

# training
init_op = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init_op)
batch_size = 100
no_of_batches = int(len(train_input)) / batch_size
epoch = 5000
for i in range(epoch):
    ptr = 0
    for j in range(no_of_batches):
        inp, out = train_input[ptr:ptr+batch_size], train_output[ptr:ptr+batch_size]
        ptr+=batch_size
        sess.run(minimize,{data: inp, target: out})
    print "Epoch ",str(i)

# test
incorrect = sess.run(error,{data: test_input, target: test_output})
print sess.run(prediction,{data: [[[1],[0],[0],[1],[1],[0],[1],[1],[1],[0],[1],[0],[0],[1],[1],[0],[1],[1],[1],[0]]]})
print('Epoch {:2d} error {:3.1f}%'.format(i + 1, 100 * incorrect))
sess.close()
