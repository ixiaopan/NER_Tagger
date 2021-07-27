import os
from shutil import copy
import torch
from utils import utils

from batch_train_baseline import train_and_evaluate
from models.BiLSTM_batch import BiLSTM_CRF_Batch

def main():
  corpus_summary = utils.read_json('./data/data_summary.json')
 
  sorted_domain_ratio = sorted([
    (name, float(domain['total_tokens_with_ne_ratio']))
    for name, domain in corpus_summary['genres'].items() if name != 'pt'
  ], key=lambda x: x[1], reverse=True)

  sorted_domains = [ name for name, _ in sorted_domain_ratio ]


  # parameter transfer
  pre_domain = None
  model_param_dir = './experiments/pool_init'
  # shared_dict = ['char_embed', 'char_lstm', 'word_embed', 'dropout', 'lstm']
  for name in sorted_domains: # for each domain
    train_data_dir = './data/' + name

    print('Training {}...'.format(train_data_dir))

    # prepare model
    model, params, embedding_params_dir = utils.init_baseline_model(
      BiLSTM_CRF_Batch,
      model_param_dir,
      train_data_dir
    )

    # initialize model from the previous model
    # './experiments/pool_init/bc'
    if pre_domain is not None:
      print('INIT transfer from {} to {}...'.format(pre_domain, name))

      pre_model_dict = torch.load(os.path.join(model_param_dir, pre_domain, 'best.pth.tar'))
      pre_model_dict = pre_model_dict['model_dict']
      # ['char_embed.weight', '
      # char_lstm.weight_ih_l0', 'char_lstm.weight_hh_l0', 'char_lstm.bias_ih_l0', 'char_lstm.bias_hh_l0', 
      # 'char_lstm.weight_ih_l0_reverse', 'char_lstm.weight_hh_l0_reverse', 'char_lstm.bias_ih_l0_reverse', 'char_lstm.bias_hh_l0_reverse', 
      # 'word_embed.weight', 
      # 'lstm.weight_ih_l0', 'lstm.weight_hh_l0', 'lstm.bias_ih_l0', 'lstm.bias_hh_l0', '
      # lstm.weight_ih_l0_reverse', 'lstm.weight_hh_l0_reverse', 'lstm.bias_ih_l0_reverse', 'lstm.bias_hh_l0_reverse', 
      # 'fc.weight', 'fc.bias', 
      # 'crf.transitionMatrix', 'crf.start_transitions', 'crf.stop_transitions']
      # pre_model_dict = {k: v for k, v in pre_model_dict.items() if k.split('.')[0] in shared_dict}

      # cur_model_dict = model.state_dict()
      # cur_model_dict.update(pre_model_dict) 
      model.load_state_dict(pre_model_dict)

    pre_domain = name

    train_and_evaluate(
      train_data_dir, 
      model_param_dir=model_param_dir, 
      model_meta={ 'model': model, 'params': params, 'embedding_params_dir': embedding_params_dir }
    )
  


  print('copy', sorted_domains[-1], 'into pool_init...')
  dest_dir = os.path.join(model_param_dir, 'pool_init')
  if not os.path.exists(dest_dir):
    os.mkdir(dest_dir)

  copy(os.path.join(model_param_dir, sorted_domains[-1], 'best.pth.tar'), dest_dir)


if __name__ == '__main__':
  main()
