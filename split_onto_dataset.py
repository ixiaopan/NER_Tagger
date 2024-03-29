import os
import argparse
import numpy as np

from utils import utils

parser = argparse.ArgumentParser()
parser.add_argument('--domain', help="domain name")

if __name__ == '__main__':
  args = parser.parse_args()

  print('=== split dataset into train, valid and test ===')

  pool_data = {
    'train_sentences': [],
    'train_labels': [],
    
    'valid_sentences': [],
    'valid_labels': [],
    
    'test_sentences': [],
    'test_labels': []
  }

  data_summary = utils.read_json('./data/data_summary.json')
  if args.domain:
    domains = [ args.domain ]
  else:
    domains = list(data_summary['genres'].keys())

  for type in domains:
    corpus, corpus_tag = utils.load_ner_data(os.path.join('./data', type, 'dataset.csv'))

    corpus = np.array(corpus, dtype=object)
    corpus_tag = np.array(corpus_tag, dtype=object)

    for name, X, y in utils.split_train_val_test(corpus, corpus_tag):
      if args.domain is None and type != 'pt': # we don't include 'pt'
        pool_data[name + '_sentences'] += list(X)
        pool_data[name + '_labels'] += list(y)

      utils.save_text(os.path.join('./data', type, name, 'sentences.txt'), X, lambda s: ' '.join(s))
      utils.save_text(os.path.join('./data', type, name, 'labels.txt'), y, lambda s: ' '.join(s))
  
    print(' - {} done'.format(type))

  # all availabel training data from the above six domains
  for name in ['train', 'valid', 'test']:
    utils.save_text(os.path.join('./data/pool', name, 'sentences.txt'), pool_data[name + '_sentences'], lambda s: ' '.join(s))
    utils.save_text(os.path.join('./data/pool', name, 'labels.txt'), pool_data[name + '_labels'], lambda s: ' '.join(s))
  print(' - pool done')
