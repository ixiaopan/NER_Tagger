#!/bin/bash

data_dir='./data/'$1
model_param_dir='./experiments/pool_'$1
max_len=$2

python batch_eval_baseline.py --data_dir=$data_dir --split_type='test_sent_10' --model_param_dir=$model_param_dir
python batch_eval_baseline.py --data_dir=$data_dir --split_type='test_sent_30' --model_param_dir=$model_param_dir
python batch_eval_baseline.py --data_dir=$data_dir --split_type='test_sent_60' --model_param_dir=$model_param_dir
python batch_eval_baseline.py --data_dir=$data_dir --split_type='test_sent_'$max_len --model_param_dir=$model_param_dir


python batch_eval_baseline.py --data_dir=$data_dir --split_type='test_rare_7' --model_param_dir=$model_param_dir
python batch_eval_baseline.py --data_dir=$data_dir --split_type='test_rare_12' --model_param_dir=$model_param_dir
python batch_eval_baseline.py --data_dir=$data_dir --split_type='test_rare_25' --model_param_dir=$model_param_dir
python batch_eval_baseline.py --data_dir=$data_dir --split_type='test_rare_50' --model_param_dir=$model_param_dir
python batch_eval_baseline.py --data_dir=$data_dir --split_type='test_rare_100' --model_param_dir=$model_param_dir
python batch_eval_baseline.py --data_dir=$data_dir --split_type='test_rare_common' --model_param_dir=$model_param_dir
