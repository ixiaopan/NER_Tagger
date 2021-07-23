#!/bin/bash

data_dir='./data/'$1

python batch_train_baseline.py --train_data_dir=$data_dir

python batch_eval_baseline.py --data_dir=$data_dir
