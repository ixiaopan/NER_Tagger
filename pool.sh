#!/bin/bash

domain=$1

python batch_train_baseline.py --train_data_dir='./data/pool_'$domain --model_param_dir='./experiments/pool_'$domain

python batch_eval_baseline.py --data_dir='./data/'$domain --model_param_dir='./experiments/pool_'$domain
