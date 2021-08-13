import os
import numpy as np
from utils import utils

if __name__ == '__main__':
  print('=== leave-one-domain-out ===')

  domains = [ 'bc', 'bn', 'nw', 'mz', 'tc', 'wb' ]

  for cur_genre in domains:
    remaining_domains = [ d for d in domains if d != cur_genre]

    # keep validation set
    for dtype in ['valid']:
      X = []
      y = []

      for other_domain in remaining_domains:
        d_sent = utils.read_text(os.path.join('./data', other_domain, dtype, 'sentences.txt'))
        X += d_sent

        d_label = utils.read_text(os.path.join('./data', other_domain, dtype, 'labels.txt'))
        y += d_label

      utils.save_text(os.path.join('./data', 'pool_' + cur_genre, dtype, 'sentences.txt'), X)
      utils.save_text(os.path.join('./data', 'pool_' + cur_genre, dtype, 'labels.txt'), y)


    # merge 'train&test'
    X = []
    y = []
    for dtype in ['train']:
      for other_domain in remaining_domains:
        d_sent = utils.read_text(os.path.join('./data', other_domain, dtype, 'sentences.txt'))
        X += d_sent

        d_label = utils.read_text(os.path.join('./data', other_domain, dtype, 'labels.txt'))
        y += d_label

    utils.save_text(os.path.join('./data', 'pool_' + cur_genre, 'train', 'sentences.txt'), X)
    utils.save_text(os.path.join('./data', 'pool_' + cur_genre, 'train', 'labels.txt'), y)

    print(' - {} done'.format(cur_genre))
