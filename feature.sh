#!/bin/bash


model_param_dir='./experiments/mult_private'
domains='nw tc'


# sent len
# seq_len='1,3,6,15,25,35,60,9999'
# seq_len_list='1 3 6 15 25 35 60'
# for d in $domains; do
#   python split_by_sent_len.py --domain=$d --sent_len_threshold=$seq_len
#   for l in $seq_len_list; do
#     python batch_eval_mult.py --data_dir='./data/'$d --sub_dataset='test_sent_'$l --model_param_dir=$model_param_dir
#   done
# done



# rare word
# rare_freq='1,2,4,9,29,999999'
# rare_freq_list='1 2 4 9 29'
# for d in $domains; do
#   python split_by_rare_word.py --domain=$d --rare_freq=$rare_freq
#   for l in $rare_freq_list; do
#     python batch_eval_mult.py --data_dir='./data/'$d --sub_dataset='test_rare_'$l --model_param_dir=$model_param_dir
#   done
# done



# pos - IN
# prep_len='0,1,3,4,6,999'
# prep_len_list='0 1 3 4 6'
# for d in $domains; do
#   python split_by_pos.py --domain=$d --pos='IN-TO' --breakpoint=$prep_len
#   for l in $prep_len_list; do
#     python batch_eval_mult.py --data_dir='./data/'$d --sub_dataset='test_pos_prep_'$l --model_param_dir=$model_param_dir
#   done
# done



verb_len='0,1,3,4,6,999'
verb_len_list='0 1 3 4 6'
for d in $domains; do
  python split_by_pos.py --domain=$d --pos='VB-VBD-VBG-VBN-VBP-VBZ' --breakpoint=$verb_len
  for l in $verb_len_list; do
    python batch_eval_mult.py --data_dir='./data/'$d --sub_dataset='test_pos_verb_'$l --model_param_dir=$model_param_dir
  done
done
