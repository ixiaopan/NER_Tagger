import os
import csv
import argparse
import numpy as np
from utils import utils

def load_ner_data(filename=None, encoding='windows-1252'):
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



def main(data_dir, filename, encoding):
  print('=== Loading dataset from {} ==='.format(data_dir))
  corpus, corpus_tag = load_ner_data(os.path.join(data_dir, filename), encoding)
  print('Total sentences, ', len(corpus))
  print('Total sentences tag, ', len(corpus_tag))



  print('=== Split & Save dataset ===')
  corpus = np.array(corpus, dtype=object)
  corpus_tag = np.array(corpus_tag, dtype=object)
  for name, X, y in utils.split_train_val_test(corpus, corpus_tag):
    utils.save_text(os.path.join(data_dir, name, 'sentences.txt'), X, lambda s: ' '.join(s))
    utils.save_text(os.path.join(data_dir, name, 'labels.txt'), y, lambda s: ' '.join(s))
    print(name, 'done')


  print('=== Build dataset done ===')



parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='./data/kaggle', help="Directory containing the dataset")
parser.add_argument('--filename', default='ner_dataset.csv', help="filename")
parser.add_argument('--encoding', default='windows-1252', help="encoding")

if __name__ == '__main__':
  args = parser.parse_args()
  main(args.data_dir, args.filename, args.encoding)
