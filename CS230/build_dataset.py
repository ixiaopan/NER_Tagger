"""
Split raw dataset  into train(70%), valid(10%), and test(20%) data set
"""

import os
import csv
import numpy as np
import argparse

from utils import utils

def load_ner_data(filename=None, encoding='utf-8'):
  msg = filename + ' is not found.'
  assert os.path.isfile(filename), msg

  with open(filename, encoding=encoding) as f:
    csvreader = csv.reader(f, delimiter=',')

    corpus = []
    corpus_tag = []
    sentence = []
    sentence_tag = []
    for index, row in enumerate(csvreader):
      if index == 0: # header
        continue

      sentence_flag, word, pos, chunk_tag = row

      if len(sentence_flag) > 0: # start from a new sentence        
        sentence = []
        sentence_tag = []

        corpus.append(sentence)
        corpus_tag.append(sentence_tag)

      sentence.append(word)
      sentence_tag.append(chunk_tag)

    return corpus, corpus_tag


def split_train_val_test(corpus, corpus_tag):
  corpus = np.array(corpus, dtype=object)
  corpus_tag = np.array(corpus_tag, dtype=object)
  
  total_len = len(corpus)
  train_cutoff = int(0.7 * total_len)
  valid_cutoff = int(0.8 * total_len)

  rand_idx = np.random.permutation(total_len)
  train_idx = rand_idx[:train_cutoff]
  valid_idx = rand_idx[train_cutoff:valid_cutoff]
  test_idx =  rand_idx[valid_cutoff:]

  X_train, y_train = corpus[train_idx], corpus_tag[train_idx]
  X_valid, y_valid = corpus[valid_idx], corpus_tag[valid_idx]
  X_test, y_test = corpus[test_idx], corpus_tag[test_idx]

  return [('train', X_train, y_train), ('valid', X_valid, y_valid), ('test', X_test, y_test)]


def save_dataset(X, y, filepath):
  utils.save_text(os.path.join(filepath, 'sentences.txt'), X, lambda s: ' '.join(s))
  utils.save_text(os.path.join(filepath, 'labels.txt'), y, lambda s: ' '.join(s))


def main(data_dir, filename, encoding):
  print('=== Loading dataset from {} ==='.format(data_dir))
  corpus, corpus_tag = load_ner_data(os.path.join(data_dir, filename), encoding)
  print('Total sentences, ', len(corpus))
  print('Total sentences tag, ', len(corpus_tag))

  print('=== Split & Save dataset ===')
  for name, X, y in split_train_val_test(corpus, corpus_tag):
    save_dataset(X, y, os.path.join(data_dir, name))
    print(name, 'done')

  print('=== Build dataset done ===')


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='./data/kaggle', help="Directory containing the dataset")
parser.add_argument('--filename', default='ner_dataset.csv', help="filename")
parser.add_argument('--encoding', default='windows-1252', help="encoding")

if __name__ == '__main__':
  args = parser.parse_args()
  main(args.data_dir, args.filename, args.encoding)
