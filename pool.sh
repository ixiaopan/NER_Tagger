#!/bin/bash

# domain=$1

# python batch_train_baseline.py --train_data_dir='./data/pool_'$domain --model_param_dir='./experiments/pool_'$domain

# python batch_eval_baseline.py --data_dir='./data/'$domain --model_param_dir='./experiments/pool_'$domain



python batch_train_baseline.py --train_data_dir='./data/pool_bc' --model_param_dir='./experiments/pool_bc'
python batch_eval_baseline.py --data_dir='./data/bc' --model_param_dir='./experiments/pool_bc'

python batch_train_baseline.py --train_data_dir='./data/pool_bn' --model_param_dir='./experiments/pool_bn'
python batch_eval_baseline.py --data_dir='./data/bn' --model_param_dir='./experiments/pool_bn'

python batch_train_baseline.py --train_data_dir='./data/pool_mz' --model_param_dir='./experiments/pool_mz'
python batch_eval_baseline.py --data_dir='./data/mz' --model_param_dir='./experiments/pool_mz'

python batch_train_baseline.py --train_data_dir='./data/pool_nw' --model_param_dir='./experiments/pool_nw'
python batch_eval_baseline.py --data_dir='./data/nw' --model_param_dir='./experiments/pool_nw'

python batch_train_baseline.py --train_data_dir='./data/pool_tc' --model_param_dir='./experiments/pool_tc'
python batch_eval_baseline.py --data_dir='./data/tc' --model_param_dir='./experiments/pool_tc'

python batch_train_baseline.py --train_data_dir='./data/pool_wb' --model_param_dir='./experiments/pool_wb'
python batch_eval_baseline.py --data_dir='./data/wb' --model_param_dir='./experiments/pool_wb'
