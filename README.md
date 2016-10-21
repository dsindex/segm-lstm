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
  - tensorflow version
    - 0.9

- sketch code
```shell
$ python sketch.py
...
step : 970,cost : 0.0117462
step : 980,cost : 0.0115485
step : 990,cost : 0.0113553
out = 이것을 띄어쓰기하면 어떻게 될까요
out = 아버지가 방에 들어가신다.
```

- how to handle variable-length input
```protosame
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
```shell
$ python train.py --train=train.txt --validation=validation.txt --model=model --iters=30

$ python inference.py --model=model < test.txt
...
model restored from model/segm.ckpt
out = 이것을 띄어 쓰기하면 어 떻게 될까요.
out = 아버지가 방에 들어 가신다.
out = SK이노베이션, GS, S-Oil, 대림산업, 현대중공업 등 대규모 적자를 내던
out = 기업들이 극한 구조조정을 통해 흑자로 전환하거나
out = 적자폭을 축소한 것이영 업이익 개선을 이끈 것으로 풀이된다.


$ python train.py --train=big.txt --validation=validation.txt --model=model --iters=30

$ python inference.py --model=model < test.txt
out = 이것을 띄어쓰기하면 어떻게 될 까요.
out = 아버지가 방에 들어 가 신다.
out = SK이노베이션, GS, S-Oil,대림산업, 현대 중공업등대규모적자를 내던
out = 기업들이 극한 구조조정 을 통해 흑자로 전환하거나
out = 적자폭을 축소한 것이 영업이 익개선을 이 끈것으로 풀이 된 다.

# it seems that training data is not enough...
```

- character-based word2vec
```shell
# word2vec : https://github.com/tensorflow/tensorflow/tree/master/tensorflow/models/embedding
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
```shell
$ python train_emb.py --train=big.txt --validation=validation.txt --embedding=emb --model=model_emb --iters=30

$ python inference_emb.py -e emb -m model_emb < test.txt
out = 이것을 띄어쓰기하면 어떻게 될 까요.
out = 아버지가 방에 들어가 신다.
out = SK이노베이션, GS, S-Oil, 대림산업, 현대중공업등대규모적자를 내던
out = 기업들이 극한 구조조정을 통해 흑자로 전환하거나
out = 적자폭을 축소한 것 이 영업이익개선을 이 끈것으로 풀이된 다.

# prepare bigbig.txt(53548 news articles)
$ python train_emb.py --train=bigbig.txt --validation=validation.txt --embedding=emb --model=model_emb --iters=3
...
53545 th sentence ... done
53546 th sentence ... done
53547 th sentence ... done
seq : 2,validation cost : 7.31046978633,validation accuracy : 0.905555615822
save model(final)
end of training

# it takes 3 days long. ;;

$ python inference_emb.py -e emb -m model_emb < test.txt
out = 이것을 띄어쓰기하면 어떻게 될 까요.
out = 아버지가 방에 들어가 신다.
out = SK 이 노베이션, GS, S-Oil, 대림산업, 현대중공업등대규모적자를 내던
out = 기업들이 극한 구조조정을 통해 흑자로 전환하거나
out = 적자폭을 축소한 것이 영업이 익개선을 이 끈것으로 풀이 된다.

$ python inference_emb.py -e emb -m model_emb
유치원음악회가열리는날입니다.
out = 유치원음악회가 열리는 날 입니다.
친구들은커서무엇이되고싶습니까
out = 친구들은 커서 무엇이 되고 싶습니까
```

- development note
```protosame
- training speed is very slow despite of using GPU. 
  how make it faster?
  - increasing batch_size
    we need some tricky code works that process file to generate batch using `yield`
  - increasing number of threads
  - using distributed training
- tuning points
  - trained model from news corpus seems to be weak for verbal words. so we need to prepare a verbal corpus from somewhere.
    - ex) '날이에요','싶나요','해요'
  - iterations
  - hidden layer dimension
  - embedding dimension
- when train_emb.py is running, it is not possible to run train.py simultaneously.
  we need to figure out.
```

- references
  - [using a pretrained word embedding](https://codedump.io/share/GsajBJMQJ50P/1/using-a-pre-trained-word-embedding-word2vec-or-glove-in-tensorflow)
