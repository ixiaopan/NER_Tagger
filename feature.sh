#!/bin/bash


model_param_dir='./experiments/mult_private'
domains='nw tc'


# sent len
seq_len='2,3,6,10,20,30,60,100,9999'
seq_len_list='2 3 6 10 20 30 60 100'
for d in $domains; do
  python split_by_sent_len.py --domain=$d --sent_len_threshold=$seq_len
  for l in $seq_len_list; do
    python batch_eval_mult.py --data_dir='./data/'$d --sub_dataset='test_sent_'$l --model_param_dir=$model_param_dir
  done
done



# rare word
# rare_freq='1,2,3,4,5,6,9,26,999999'
# rare_freq_list='1 2 3 4 5 6 9 26'
# for d in $domains; do
#   python split_by_rare_word.py --domain=$d --rare_freq=$rare_freq
#   for l in $rare_freq_list; do
#     python batch_eval_mult.py --data_dir='./data/'$d --sub_dataset='test_rare_'$l --model_param_dir=$model_param_dir
#   done
# done



# pos - IN
# prep_len='0,2,4,6,10,9999'
# prep_len_list='0 2 4 6 10'
# for d in $domains; do
#   python split_by_pos.py --domain=$d --pos='IN' --breakpoint=$prep_len
#   for l in $prep_len_list; do
#     python batch_eval_mult.py --data_dir='./data/'$d --sub_dataset='test_pos_prep_'$l --model_param_dir=$model_param_dir
#   done
# done
