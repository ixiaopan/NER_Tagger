import os
from shutil import copy

from utils import utils
from train_baseline import train_and_evaluate


def main():
  corpus_summary = utils.read_json('./data/data_summary.json')
 
  sorted_domain_ratio = sorted([
    (name, float(domain['total_tokens_with_ne_ratio']))
    for name, domain in corpus_summary['genres'].items() if name != 'pt'
  ], key=lambda x: x[1], reverse=True)
 
  sorted_domains = [ name for name, _ in sorted_domain_ratio ]

  # transfer learning
  model_weight_filepath = None
  model_param_dir = './experiments/pool_init'
  for name in sorted_domains:
    train_data_dir = os.path.join('./data', name)

    train_and_evaluate(
      train_data_dir, 
      model_param_dir=model_param_dir, 
      best_metric='micro_f1',
      model_weight_filepath=model_weight_filepath
    )

    # './experiments/pool_init/bc'
    model_weight_filepath = os.path.join(model_param_dir, name, 'best.pth.tar')


  # 
  print('copy the last model into pool_init...')
  dest_dir = os.path.join(model_param_dir, 'pool_init')
  if not os.path.exists(dest_dir):
    os.makedir(dest_dir)

  copy(os.path.join(model_param_dir, sorted_domains[-1], 'best.pth.tar'), dest_dir)


if __name__ == '__main__':
  main()
