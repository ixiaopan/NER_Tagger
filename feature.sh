#!/bin/bash

# data_dir='./data/'$1
# model_param_dir='./experiments/pool_'$1
model_param_dir='./experiments/mult_private'

# sent len
python split_by_sent_len.py --domain='nw'
python batch_eval_mult.py --data_dir='./data/nw' --sub_dataset='test_sent_2' --model_param_dir=$model_param_dir
python batch_eval_mult.py --data_dir='./data/nw' --sub_dataset='test_sent_6' --model_param_dir=$model_param_dir
python batch_eval_mult.py --data_dir='./data/nw' --sub_dataset='test_sent_10' --model_param_dir=$model_param_dir
python batch_eval_mult.py --data_dir='./data/nw' --sub_dataset='test_sent_58' --model_param_dir=$model_param_dir

python split_by_sent_len.py --domain='tc'
python batch_eval_mult.py --data_dir='./data/tc' --sub_dataset='test_sent_2' --model_param_dir=$model_param_dir
python batch_eval_mult.py --data_dir='./data/tc' --sub_dataset='test_sent_6' --model_param_dir=$model_param_dir
python batch_eval_mult.py --data_dir='./data/tc' --sub_dataset='test_sent_10' --model_param_dir=$model_param_dir
python batch_eval_mult.py --data_dir='./data/tc' --sub_dataset='test_sent_58' --model_param_dir=$model_param_dir


# rare word
# python batch_eval_mult.py --data_dir=$data_dir --sub_dataset='test_rare_7' --model_param_dir=$model_param_dir
# python batch_eval_mult.py --data_dir=$data_dir --sub_dataset='test_rare_12' --model_param_dir=$model_param_dir
# python batch_eval_mult.py --data_dir=$data_dir --sub_dataset='test_rare_25' --model_param_dir=$model_param_dir
# python batch_eval_mult.py --data_dir=$data_dir --sub_dataset='test_rare_50' --model_param_dir=$model_param_dir
# python batch_eval_mult.py --data_dir=$data_dir --sub_dataset='test_rare_100' --model_param_dir=$model_param_dir
# python batch_eval_mult.py --data_dir=$data_dir --sub_dataset='test_rare_common' --model_param_dir=$model_param_dir

# pos