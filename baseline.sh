#!/bin/bash

# data_dir='./data/'$1
# python batch_train_baseline.py --train_data_dir=$data_dir
# python batch_eval_baseline.py --data_dir=$data_dir

python batch_train_baseline.py --train_data_dir='./data/bc'
python batch_eval_baseline.py --data_dir='./data/bc'

python batch_train_baseline.py --train_data_dir='./data/bn'
python batch_eval_baseline.py --data_dir='./data/bn'

python batch_train_baseline.py --train_data_dir='./data/mz'
python batch_eval_baseline.py --data_dir='./data/mz'

python batch_train_baseline.py --train_data_dir='./data/nw'
python batch_eval_baseline.py --data_dir='./data/nw'

python batch_train_baseline.py --train_data_dir='./data/tc'
python batch_eval_baseline.py --data_dir='./data/tc'

python batch_train_baseline.py --train_data_dir='./data/wb'
python batch_eval_baseline.py --data_dir='./data/wb'
