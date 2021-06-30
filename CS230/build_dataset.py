import os
import argparse
import numpy as np
from utils import utils


def main(data_dir, filename, encoding):
  print('Loading dataset from {}'.format(data_dir))
  corpus, corpus_tag = utils.load_ner_data(os.path.join(data_dir, filename), encoding)

  print('Split & Save dataset')
  corpus = np.array(corpus, dtype=object)
  corpus_tag = np.array(corpus_tag, dtype=object)
 
  for name, X, y in utils.split_train_val_test(corpus, corpus_tag):
    utils.save_text(os.path.join(data_dir, name, 'sentences.txt'), X, lambda s: ' '.join(s))
    utils.save_text(os.path.join(data_dir, name, 'labels.txt'), y, lambda s: ' '.join(s))
 
    print(' -', name, 'done')

  print('Build dataset done')


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='./data/kaggle', help="Directory containing the dataset")
parser.add_argument('--filename', default='ner_dataset.csv', help="filename")
parser.add_argument('--encoding', default='windows-1252', help="encoding")

if __name__ == '__main__':
  args = parser.parse_args()
  main(args.data_dir, args.filename, args.encoding)
