segm-lstm
===

- description
  - string segmentation(auto-spacing) using LSTM(tensorflow)
    - input
      - string, ex) '이것을띄어쓰기하면어떻게될까요'
    - output
      - string, ex) '이것을 띄어쓰기하면 어떻게 될까요' 
  - model
    - x : '이것을 띄어쓰기하면 어떻게 될까요'
	- y : '0 0 1 0 0 0 0 0 0 1 0 0 0 1 0 0 0 0'
	  - 1 : if next char is space
	  - 0 : if next char is not space
    - learn to predict tag sequence

- sketch code
```
$ python sketch.py
...
step : 970,cost : 0.0117462
step : 980,cost : 0.0115485
step : 990,cost : 0.0113553
out = 이것을 띄어쓰기하면 어떻게 될까요
out = 아버지가 방에 들어가신다.
```

- how to handle variable-length input
```
let's try to use sliding window method and early stop.

n_steps = 30

- training
  if len(sentence) >= 1 and len(sentence) < n_steps : padding with '\t'
  if len(sentence) > n_steps : move next batch pointer(sliding window)

- inference
  if len(sentence) >= 1 and len(sentence) < n_steps : padding with '\t'
  if len(sentence) > n_steps : 
    move next batch pointer(sliding window)
	merge result into one array
	decoding
```

- train and inference
```
$ python train.py --train=train.txt --validation=validation.txt --model=model

$ python inference.py --model=model < test.txt
...
model restored from model/segm.ckpt
out = 이것을 띄어 쓰기하면 어 떻게 될까요.
out = 아버지가 방에 들어 가신다.
out = SK이노베이션, GS, S-Oil, 대림산업, 현대중공업 등 대규모 적자를 내던
out = 기업들이 극한 구조조정을 통해 흑자로 전환하거나
out = 적자폭을 축소한 것이영 업이익 개선을 이끈 것으로 풀이된다.


$ python train.py --train=big.txt --validation=validation.txt --model=model
...
7 th sentence ... done
8 th sentence ... done
9 th sentence ... done
seq : 29,validation cost : 124.562777519,validation accuracy : 0.942500010133
save dic
save model(final)
end of training

$ python inference.py --model=model < test.txt
...
model restored from model/segm.ckpt
out = 이것 을 띄어쓰기 하면어 떻게 될 까 요.
out = 아버 지 가방에들 어가 신다 .
out = SK이노베이 션, GS , S -Oil, 대림산 업, 현대중공 업등 대규모적자 를 내 던
out = 기업들이 극 한 구조조정을 통해흑자로 전환하거나
out = 적자 폭을 축소한 것 이 영업이익 개선을 이 끈 것 으로 풀이 된 다.

# it seems that training data is not enough...
```

- character-based word2vec
```
# usage : https://github.com/tensorflow/tensorflow/tree/master/tensorflow/models/embedding
# modify save_vocab() to train non-ascii data
# modify eval() to avoid zero-devision
$ cp tensorflow/tensorflow/models/embedding/word2vec_optimized.py .
$ vi word2vec_optimized.py
  ...
  def save_vocab(self):
    """Save the vocabulary to a file so the model can be reloaded."""
    opts = self._options
    with open(os.path.join(opts.save_path, "vocab.txt"), "w") as f:
      for i in xrange(opts.vocab_size):
        f.write("%s %d\n" % (tf.compat.as_text(opts.vocab_words[i]).encode('utf-8'),
                             opts.vocab_counts[i]))
  ...
  def eval(self):
  ...
  print("Eval %4d/%d accuracy = %4.1f%%" % (correct, total,
                                              0 if total is 0 else correct * 100.0 / total))
  ...
# preprocessing for character-based
$ python tochar.py < bigbig.txt > bigbig.txt.char

# train word2vec
$ mkdir emb
$ python word2vec_optimized.py --train_data=bigbig.txt.char --eval_data=questions-words.txt --embedding_size=200 --save_path=emb

# test word2vec
$ cd segm-lstm
$ python test_word2vec.py --embedding_size=200 --model_path=emb
...
가
=====================================
가                  1.0000
감                  0.9716
알                  0.9695
니                  0.9681
기                  0.9680
런                  0.9659
쥬                  0.9640
...

# you can dump embedding by using embedding_dump() in test_word2vec.py
$ python test_word2vec.py --embedding_size=200 --model_path=emb --embedding_dump=1
# now you have embeddings data in emb/embedding.pickle

```

- train and inference with character embedding
```
$ python train_emb.py --train=big.txt --validation=validation.txt --embedding=emb --model=model_emb

$ python inference_emb.py -e emb -m model_emb < test.txt

```

- development note
```
- training speed is very slow despite of using GPU. 
  how make it faster? what about using word2vec(character-based)?
  and more batch_size?
- increasing batch_size
  some tricky code works are needed
- using a pretrained word embedding
  https://codedump.io/share/GsajBJMQJ50P/1/using-a-pre-trained-word-embedding-word2vec-or-glove-in-tensorflow
```
