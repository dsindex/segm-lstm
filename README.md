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

- how to deal variable-length input
```
let's try to use sliding window method and early stop.

n_steps = 20

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

- train
```
$ python segm_train.py --train=train.txt --model=model
...
(i,seq) : (2,20),cost : 2.24725
(i,seq) : (2,30),cost : 1.8682
(i,seq) : (2,40),cost : 1.5787

이것을띄어쓰기하면어떻게될까요
out = 이것을 띄어 쓰기하면 어떻게 될까요
아버지가방에들어가신다
out = 아버지가 방에 들어 가신다
기업들이극한구조조정을통해
out = 기업들이 극한 구조조정을 통해
```

- inference
```

```
