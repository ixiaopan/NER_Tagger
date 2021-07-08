import os
import argparse
import numpy as np

from utils import utils

parser = argparse.ArgumentParser()
parser.add_argument('--domain', help="domain name")

if __name__ == '__main__':
  args = parser.parse_args()

  print('=== split dataset into train, valid and test ===')

  data_summary = utils.read_json('./data/data_summary.json')
  if args.domain:
    domains = [ args.domain ]
  else:
    domains = list(data_summary['genres'].keys()) + ['pool']

  for type in domains:
    corpus, corpus_tag = utils.load_ner_data(os.path.join('./data', type, 'dataset.csv'))

    corpus = np.array(corpus, dtype=object)
    corpus_tag = np.array(corpus_tag, dtype=object)

    for name, X, y in utils.split_train_val_test(corpus, corpus_tag):
      utils.save_text(os.path.join('./data', type, name, 'sentences.txt'), X, lambda s: ' '.join(s))
      utils.save_text(os.path.join('./data', type, name, 'labels.txt'), y, lambda s: ' '.join(s))

    print(' - {} done'.format(type))

  print('\n=== We\'re done ===')
