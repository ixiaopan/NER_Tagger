#!/bin/bash

data_dir='./data/pool'

python build_onto_profile.py --data_dir=$data_dir

python train_baseline.py --data_dir=$data_dir

python eval_baseline.py --data_dir='./data/bc'
python eval_baseline.py --data_dir='./data/bn'
python eval_baseline.py --data_dir='./data/mz'
python eval_baseline.py --data_dir='./data/nw'
python eval_baseline.py --data_dir='./data/tc'
python eval_baseline.py --data_dir='./data/wb'
