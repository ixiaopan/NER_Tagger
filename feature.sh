#!/bin/bash

model_param_dir='./experiments/mult_private'
domains='bc nw tc'

# sent len
seq_len='2,6,10,58,9999'
seq_len_list='2 6 10 58'
for d in $domains; do
  python split_by_sent_len.py --domain=$d --sent_len_threshold=$seq_len
  for l in $seq_len_list; do
    python batch_eval_mult.py --data_dir='./data/'$d --sub_dataset='test_sent_'$l --model_param_dir=$model_param_dir
  done
done


# rare word
# python batch_eval_mult.py --data_dir=$data_dir --sub_dataset='test_rare_7' --model_param_dir=$model_param_dir
# python batch_eval_mult.py --data_dir=$data_dir --sub_dataset='test_rare_common' --model_param_dir=$model_param_dir


# pos