from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import threading
import time

from six.moves import xrange  # pylint: disable=redefined-builtin

import numpy as np
import tensorflow as tf

from tensorflow.models.embedding import gen_word2vec as word2vec

flags = tf.app.flags

flags.DEFINE_string("model_path", None, "Directory to saved model.")
flags.DEFINE_integer("embedding_size", 200, "The embedding dimension size.")

FLAGS = flags.FLAGS

class Options(object):
  """Options used by our word2vec model."""

  def __init__(self):
    # Model options.

    # Embedding dimension.
    self.emb_dim = FLAGS.embedding_size

    # Directory to saved model
    self.model_path = FLAGS.model_path

class Word2Vec(object):
  """Word2Vec model (Skipgram)."""

  def __init__(self, options, session):
    self._options = options
    self._session = session
    self._word2id = {}
    self._id2word = []
    self.load_vocab()
    self.build_eval_graph()

  def load_vocab(self):
    """Load the vocabulary from a file"""
    opts = self._options

    vocab_words = {}
    vocab_counts = {}
    idx = 0
    with open(os.path.join(opts.model_path, "vocab.txt"), "r") as f:
      for line in f :
        word, count = line.split(' ')
        vocab_words[idx] = word
        vocab_counts[idx] = count
        idx += 1
    opts.vocab_size = idx
    opts.vocab_words = vocab_words
    opts.vocab_counts = vocab_counts
    self._id2word = opts.vocab_words
    for i, w in self._id2word.iteritems():
      self._word2id[w] = i

  def build_eval_graph(self):
    """Build the evaluation graph."""
    opts = self._options

    # Declare all variables we need.
    # Input words embedding: [vocab_size, emb_dim]
    w_in = tf.Variable(
        tf.random_uniform(
            [opts.vocab_size,
             opts.emb_dim], -0.5 / opts.emb_dim, 0.5 / opts.emb_dim),
        name="w_in")
    self._w_in = w_in

    # Eval graph

    # Each analogy task is to predict the 4th word (d) given three
    # words: a, b, c.  E.g., a=italy, b=rome, c=france, we should
    # predict d=paris.

    # The eval feeds three vectors of word ids for a, b, c, each of
    # which is of size N, where N is the number of analogies we want to
    # evaluate in one batch.
    analogy_a = tf.placeholder(dtype=tf.int32)  # [N]
    analogy_b = tf.placeholder(dtype=tf.int32)  # [N]
    analogy_c = tf.placeholder(dtype=tf.int32)  # [N]

    # Normalized word embeddings of shape [vocab_size, emb_dim].
    nemb = tf.nn.l2_normalize(self._w_in, 1)

    # Each row of a_emb, b_emb, c_emb is a word's embedding vector.
    # They all have the shape [N, emb_dim]
    a_emb = tf.gather(nemb, analogy_a)  # a's embs
    b_emb = tf.gather(nemb, analogy_b)  # b's embs
    c_emb = tf.gather(nemb, analogy_c)  # c's embs

    # We expect that d's embedding vectors on the unit hyper-sphere is
    # near: c_emb + (b_emb - a_emb), which has the shape [N, emb_dim].
    target = c_emb + (b_emb - a_emb)

    # Compute cosine distance between each pair of target and vocab.
    # dist has shape [N, vocab_size].
    dist = tf.matmul(target, nemb, transpose_b=True)

    # For each question (row in dist), find the top 4 words.
    _, pred_idx = tf.nn.top_k(dist, 4)

    # Nodes for computing neighbors for a given word according to
    # their cosine distance.
    nearby_word = tf.placeholder(dtype=tf.int32)  # word id
    nearby_emb = tf.gather(nemb, nearby_word)
    nearby_dist = tf.matmul(nearby_emb, nemb, transpose_b=True)
    nearby_val, nearby_idx = tf.nn.top_k(nearby_dist,
                                         min(1000, opts.vocab_size))

    # Nodes in the construct graph which are used by training and
    # evaluation to run/feed/fetch.
    self._analogy_a = analogy_a
    self._analogy_b = analogy_b
    self._analogy_c = analogy_c
    self._analogy_pred_idx = pred_idx
    self._nearby_word = nearby_word
    self._nearby_val = nearby_val
    self._nearby_idx = nearby_idx

    # Properly initialize all variables.
    tf.initialize_all_variables().run()

    # Embedding lookup
    self._nemb = nemb
    self._nearby_emb = nearby_emb

    self.saver = tf.train.Saver()

  def _predict(self, analogy):
    """Predict the top 4 answers for analogy questions."""
    idx, = self._session.run([self._analogy_pred_idx], {
        self._analogy_a: analogy[:, 0],
        self._analogy_b: analogy[:, 1],
        self._analogy_c: analogy[:, 2]
    })
    return idx

  def analogy(self, w0, w1, w2):
    """Predict word w3 as in w0:w1 vs w2:w3."""
    wid = np.array([[self._word2id.get(w, 0) for w in [w0, w1, w2]]])
    idx = self._predict(wid)
    for c in [self._id2word[i] for i in idx[0, :]]:
      if c not in [w0, w1, w2]:
        return c
    return "unknown"

  def nearby(self, words, num=20):
    """Prints out nearby words given a list of words."""
    ids = np.array([self._word2id.get(x, 0) for x in words])
    vals, idx = self._session.run(
        [self._nearby_val, self._nearby_idx], {self._nearby_word: ids})
    for i in xrange(len(words)):
      print("\n%s\n=====================================" % (words[i]))
      for (neighbor, distance) in zip(idx[i, :num], vals[i, :num]):
        print("%-20s %6.4f" % (self._id2word[neighbor], distance))

  def embedding_lookup(self, words):
    """Get word embedding list given a list of words."""
    ids = np.array([self._word2id.get(x, 0) for x in words])
    for x in words:
      '''
      print('word = %s' % x)
      id = self._word2id.get(x, 0)
      print('id = %s' % id)
      word = self._id2word.get(id, 'UNK')
      print('word = %s' % word)
      '''
    embeddings = self._session.run(self._nearby_emb, {self._nearby_word:ids})
    for i in xrange(len(words)):
      print("\n%s\n=====================================" % (words[i]))
      print(embeddings[i])
  
  def embedding_dump(self):
    """Dump word embeddings"""
    nemb = self._session.run(self._nemb)
    for i, emb in enumerate(nemb):
      print(i, emb)

def main(_):
  """Test a word2vec model."""
  if not FLAGS.model_path:
    print("--model_path must be specified.")
    sys.exit(1)
  opts = Options()
  with tf.Graph().as_default(), tf.Session() as session:
    with tf.device("/cpu:0"):
      model = Word2Vec(opts, session)
      checkpoint_dir = opts.model_path
      ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
      if ckpt and ckpt.model_checkpoint_path :
        model.saver.restore(session, ckpt.model_checkpoint_path)
        sys.stderr.write("model restored from %s\n" %(ckpt.model_checkpoint_path))
      else :
        sys.stderr.write("no checkpoint found" + '\n')
        sys.exit(-1)
      c = model.analogy(b'france', b'paris', b'russia')
      print("analogy = %s" % c)
      model.nearby([b'france', b'paris', b'russia'])
      model.embedding_lookup([b'france', b'paris', b'russia'])
      #model.embedding_dump()
      while 1:
        try : line = sys.stdin.readline()
        except KeyboardInterrupt : break
        if not line : break
        line = line.strip()
        word = line.split()[0]
        model.nearby([word])
        model.embedding_lookup([word])

if __name__ == "__main__":
  tf.app.run()
