"""
Build vocabulary & label and summary the statistics
"""

from os import path
import argparse
from collections import Counter
from utils import utils


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='./data/toy', help="Directory containing the dataset")


PAD_CHAR = '<pad>' # fill the space for a sentence with unequal length
UNK_CHAR = 'UNK' # for word that is not present in the vocabulary
PAD_TAG = 'O'


def build_data_profile(data_dir):
  data_statistics = {
    'train_sentence_len': 0,
    'valid_sentence_len': 0,
    'test_sentence_len': 0,

    'train_tag_sentence_len': 0,
    'valid_tag_sentence_len': 0,
    'test_tag_sentence_len': 0,

    'vocab_size': 0,
    'tag_size': 0,
    'pad_word': PAD_CHAR,
    'pad_tag': PAD_TAG,
    'unk_word': UNK_CHAR
  }
  
  print('=== Build vocabulary from ', data_dir, ' ===' )
  
  # step 1
  word_counter = Counter()
  for name in ['train', 'valid', 'test']:
    word_counter, sent_len = utils.build_vocabulary(path.join(data_dir, name, 'sentences.txt'), word_counter)
    data_statistics[name + '_sentence_len'] = sent_len

  tag_counter = Counter()
  for name in ['train', 'valid', 'test']:
    tag_counter, tag_sent_len = utils.build_vocabulary(path.join(data_dir, name, 'labels.txt'), tag_counter)
    data_statistics[name + '_tag_sentence_len'] = tag_sent_len

  # should have the same number of lines
  for name in ['train', 'valid', 'test']:
    assert data_statistics[name + '_sentence_len'] == data_statistics[name + '_tag_sentence_len']

  # step 2
  vocab = [ w for w, _ in word_counter.items() ]
  tags = [ t for t, _ in tag_counter.items() ]
  if PAD_CHAR not in vocab:
    vocab.append(PAD_CHAR)
  
  if PAD_TAG not in tags:
    tags.append(PAD_TAG)

  vocab.append(UNK_CHAR)

  data_statistics['vocab_size'] = len(vocab)
  data_statistics['tag_size'] = len(tags)

  # step 3
  word_id, inverse_word_id = utils.map_word_id(vocab)
  tag_id, inverse_tag_id = utils.map_word_id(tags)

  # step 4 save meta data to file
  utils.save_text(path.join(data_dir, 'vocabulary.txt'), vocab)
  utils.save_text(path.join(data_dir, 'tags.txt'), tags)
  
  utils.save_json(path.join(data_dir, 'word_id.json'), word_id)
  utils.save_json(path.join(data_dir, 'id_word.json'), inverse_word_id)
  
  utils.save_json(path.join(data_dir, 'tag_id.json'), tag_id)
  utils.save_json(path.join(data_dir, 'inverse_tag_id.json'), inverse_tag_id)
  
  utils.save_json(path.join(data_dir, 'dataset_params.json'), data_statistics)


  # step 5: log metadata
  print('\n'.join(['{}: {}'.format(k, v) for k, v in data_statistics.items()]))
  print('=== Build vocabulary done ===')




if __name__ == '__main__':
  args = parser.parse_args()
  build_data_profile(args.data_dir)
