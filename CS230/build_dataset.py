import os
import csv
import numpy as np
from numpy.lib.npyio import save

def load_ner_data(filename='./data/kaggle/ner_dataset.csv'):
  msg = filename + 'not found.'
  assert os.path.isfile(filename), msg

  with open(filename, encoding='windows-1252') as f:
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
  corpus = np.array(corpus)
  corpus_tag = np.array(corpus_tag)
  
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
  if not os.path.exists(filepath):
    os.mkdir(filepath)

  with open(os.path.join(filepath, 'sentences.txt'), 'w') as f:
    for sent in X:
      f.write('{}\n'.format(' '.join(sent)))

  with open(os.path.join(filepath, 'labels.txt'), 'w') as f:
    for sent in y:
      f.write('{}\n'.format(' '.join(sent)))


if __name__ == '__main__':
  print('=== Loading dataset ===')
  corpus, corpus_tag = load_ner_data()
  print('Total sentences, ', len(corpus))
  print('Total sentences tag, ', len(corpus_tag))

  print('=== Save dataset ===')
  for name, X, y in split_train_val_test(corpus, corpus_tag):
    save_dataset(X, y, './data/kaggle/' + name)

  print('=== Build dataset done===')
