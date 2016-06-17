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

n_steps = 40

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
$ python train.py --train=train.txt --validation=train.txt --model=model
...
seq : 96,total cost : 2.36490138248
seq : 97,total cost : 2.32890268601
seq : 98,total cost : 2.29376351926
seq : 99,total cost : 2.25945831649
save dic
save model
end of training
...
```

- inference
```
$ python inference.py --model=model < test.txt
...
model restored from model/segm.ckpt
out = 이것을 띄어 쓰기하면 어 떻게 될까요.
out = 아버지가 방에 들어 가신다.
out = SK이노베이션, GS, S-Oil, 대림산업, 현대중공업 등 대규모 적자를 내던
out = 기업들이 극한 구조조정을 통해 흑자로 전환하거나
out = 적자폭을 축소한 것이영 업이익 개선을 이끈 것으로 풀이된다.
```
