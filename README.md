segm-lstm
===

- description
  - string segmentation using LSTM(tensorflow)
  - input
    - string, ex) '이것을띄어쓰기하면어떻게될까요'
  - output
    - string, ex) '이것을 띄어쓰기하면 어떻게 될까요' 
  - modeling function using LSTM in tensorflow

- sketch code
```
$ python segm_lstm.py
...
step : 970,cost : 0.0117462
step : 980,cost : 0.0115485
step : 990,cost : 0.0113553
out = 이것을 띄어쓰기하면 어떻게 될까요
```

