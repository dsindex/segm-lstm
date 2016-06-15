segm-lstm
===

- description
  - string segmentation using LSTM(tensorflow)
    - input
      - string, ex) '이것을띄어쓰기하면어떻게될까요'
    - output
      - string, ex) '이것을 띄어쓰기하면 어떻게 될까요' 
  - model
    - x : '이것을 띄어쓰기하면 어떻게 될까요'
	- y : '0 0 1 0 0 0 0 0 0 1 0 0 0 1 0 0 0 0'
	  - 1 : if next char is space
	  - 0 : if next char is not space
    - learn to predict next tag using LSTM

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

- test variable length lstm
```
$ python segm_lstm_vlen.py

if batch_size = 1 and n_steps depends on the size of input sentence, 
then we might think :

n_steps = tf.placeholder("int32")
x = tf.placeholder("float", [None, n_steps, n_input])
y_ = tf.placeholder("int32", [None, n_steps])

but this usage is not possible.

let's try to use sliding window method :

min_size = 5
n_steps = 50
stride = 10

- training
  if len(sentence) < min_size : continue
  if len(sentence) >= min_size and len(sentence) < n_steps : padding with ' '
  if len(sentence) > n_steps : move next batch pointer(sliding window)

- inference
  if len(sentence) < min_size : continue
  if len(sentence) >= min_size and len(sentence) < n_steps : padding with ' '
  if len(sentence) > n_steps : 
    move next batch pointer(sliding window)
	merge result into one array
	decoding
```

