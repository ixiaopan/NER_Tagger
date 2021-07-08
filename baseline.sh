#!/bin/bash

data_dir='./data/'$1

python train_baseline.py --data_dir=$data_dir

python eval_baseline.py --data_dir=$data_dir
