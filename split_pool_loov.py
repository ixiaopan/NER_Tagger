import os
import numpy as np
from utils import utils

if __name__ == '__main__':
  print('=== leave-one-domain-out ===')

  data_summary = utils.read_json('./data/data_summary.json')
  domains = [ d for d in data_summary['genres'].keys() if d != 'pt']  # we don't include 'pt'

  for cur_genre in domains:
    remaining_domains = [ d for d in domains if d != cur_genre]

    for dtype in ['train', 'valid', 'test']:
      X = []
      y = []

      for other_domain in remaining_domains:
        d_sent = utils.read_text(os.path.join('./data', other_domain, dtype, 'sentences.txt'))
        X.append(d_sent)

        d_label = utils.read_text(os.path.join('./data', other_domain, dtype, 'labels.txt'))
        y.append(d_label)

      utils.save_text(os.path.join('./data', 'pool_' + cur_genre, dtype, 'sentences.txt'), X)
      utils.save_text(os.path.join('./data', 'pool_' + cur_genre, dtype, 'labels.txt'), y)

    print(' - {} done'.format(cur_genre))

