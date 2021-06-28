"""
toolbox
"""

import os
from os import path
import json
import string
import numpy as np
from collections import Counter

import torch
from torch.utils.data import TensorDataset, DataLoader

# from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords


def save_text(filepath, text, transform_fn=None):
  parent_directory = os.path.dirname(filepath)
  if not os.path.exists(parent_directory):
    os.mkdir(parent_directory)

  with open(filepath, 'w') as f:
    for line in text:
      if transform_fn:
        line = transform_fn(line)
    
      f.write('{}\n'.format(line))



def save_json(filepath, text):
  parent_directory = os.path.dirname(filepath)
  if not os.path.exists(parent_directory):
    os.mkdir(parent_directory)


  with open(filepath, 'w') as f:
    json.dump(text, f, indent=4)


def read_json(filepath):
  with open(filepath, 'r') as f:
    data = json.load(f)

  return data


def split_train_val_test(corpus, corpus_tag, shuffle=True):
  total_len = len(corpus)
  train_cutoff = int(0.7 * total_len)
  valid_cutoff = int(0.8 * total_len)

  if shuffle:
    rand_idx = np.random.permutation(total_len)
  else:
    rand_idx = np.arange(total_len)

  train_idx = rand_idx[:train_cutoff]
  valid_idx = rand_idx[train_cutoff:valid_cutoff]
  test_idx =  rand_idx[valid_cutoff:]

  X_train, y_train = corpus[train_idx], corpus_tag[train_idx]
  X_valid, y_valid = corpus[valid_idx], corpus_tag[valid_idx]
  X_test, y_test = corpus[test_idx], corpus_tag[test_idx]

  return [
    ('train', X_train, y_train), 
    ('valid', X_valid, y_valid), 
    ('test', X_test, y_test)
  ]




def build_vocabulary(filename, word_counter=None):
  '''
  corpus: a list of sentence
  [
    'the first sentence',
    'the second sentence',
  ]
  
  return:
    - word counter
    - the number of sentences
  '''
  if word_counter is None:
    word_counter = Counter()
  
  with open(filename) as f:
    for i, line in enumerate(f.read().split('\n')):
      word_counter.update(line.split())

  return word_counter, i + 1




def map_word_id(word_list):
  word_id = {}
  inverse_word_id = {}
  
  for i, word in enumerate(word_list):
      word_id[word] = i
      inverse_word_id[i] = word
  
  return word_id, inverse_word_id




def build_ner_profile(data_dir, min_word_freq=1, min_tag_freq=1):
  '''
  build vocabulary for ner task
  '''
  PAD_CHAR = '_PAD' # fill the space for a sentence with unequal length
  UNK_CHAR = '_UNK' # for word that is not present in the vocabulary
  PAD_TAG = 'O'

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
    'unk_word': UNK_CHAR,
    'pad_tag': PAD_TAG,
  }
  
  print('=== Build NER vocabulary from ', data_dir, ' ===' )
  
  # step 1
  word_counter = Counter()
  for name in ['train', 'valid', 'test']:
    word_counter, sent_len = build_vocabulary(path.join(data_dir, name, 'sentences.txt'), word_counter)
    data_statistics[name + '_sentence_len'] = sent_len

  tag_counter = Counter()
  for name in ['train', 'valid', 'test']:
    tag_counter, tag_sent_len = build_vocabulary(path.join(data_dir, name, 'labels.txt'), tag_counter)
    data_statistics[name + '_tag_sentence_len'] = tag_sent_len

  # should have the same number of lines
  for name in ['train', 'valid', 'test']:
    assert data_statistics[name + '_sentence_len'] == data_statistics[name + '_tag_sentence_len']



  # step 2
  vocab = [ w for w, c in word_counter.items() if c >= min_word_freq ]
  tags = [ t for t, c in tag_counter.items() if c >= min_tag_freq ]
 
  if PAD_CHAR not in vocab:
    vocab.append(PAD_CHAR)

  if PAD_TAG not in tags:
    tags.append(PAD_TAG)

  vocab.append(UNK_CHAR)

  data_statistics['vocab_size'] = len(vocab)
  data_statistics['tag_size'] = len(tags)



  # step 3
  word_id, inverse_word_id = map_word_id(vocab)
  tag_id, inverse_tag_id = map_word_id(tags)



  # step 4 save meta data to file
  save_text(path.join(data_dir, 'vocabulary.txt'), vocab)
  save_text(path.join(data_dir, 'tags.txt'), tags)
  
  save_json(path.join(data_dir, 'word_id.json'), word_id)
  save_json(path.join(data_dir, 'id_word.json'), inverse_word_id)
  
  save_json(path.join(data_dir, 'tag_id.json'), tag_id)
  save_json(path.join(data_dir, 'inverse_tag_id.json'), inverse_tag_id)
  
  save_json(path.join(data_dir, 'dataset_params.json'), data_statistics)


  # step 5: log metadata
  print('\n'.join(['{}: {}'.format(k, v) for k, v in data_statistics.items()]))
  print('=== Build vocabulary done ===')



def convert_ner_dataset_to_fixed_id(data_dir, name, seq_n):
  '''
  convert each word to the corresponding id 
  '''
  data_stats = read_json(path.join(data_dir, 'dataset_params.json'))
  word_id = read_json(path.join(data_dir, 'word_id.json'))
  tag_id = read_json(path.join(data_dir, 'tag_id.json'))

  sentences, tags = [], []
  with open(path.join(data_dir, name, 'sentences.txt')) as f:
    for line in f.read().split('\n'): # for each line
      sent = []
      for w in line.split(' '):
        if w in word_id.keys():
          sent.append(word_id[w])
        else:
          sent.append(word_id[data_stats['unk_word']])
      sentences.append(sent)

  with open(path.join(data_dir, name, 'labels.txt')) as f:
    for line in f.read().split('\n'): # for each line
      tag_line = [ tag_id[t] for t in line.split(' ') ]
      tags.append(tag_line)


  assert len(sentences) == len(tags)
  for i in range(len(sentences)):
    assert len(sentences[i]) == len(tags[i])
  
  # padding, the whole dataset, it could be batch data
  if not seq_n:
    seq_n = max([ len(s) for s in sentences ])
  
  padding_sentences = word_id[data_stats['pad_word']] * np.ones((len(sentences), seq_n))
  padding_labels = -1 * np.ones((len(sentences), seq_n)) # -1 indicates padding tokens

  for i in range(len(padding_sentences)):
    cutoff = len(sentences[i])
    padding_sentences[i][:cutoff] = sentences[i]
    padding_labels[i][:cutoff] = tags[i]

  return padding_sentences, padding_labels



def build_ner_dataloader(
  data_dir, names = ['train', 'valid', 'test'], 
  seq_n = 0, batch_size = 1, shuffle=True, 
):
    '''
    data_dir: directory containing the data set
    name: ['train', 'valid', 'test']
    seq_n: fixed length of each sentence
    '''

    split = {}
    for n in ['train', 'valid', 'test']:
      if n in names:
        sentences_ids, label_ids = convert_ner_dataset_to_fixed_id(data_dir, n, seq_n)

        X, y = torch.LongTensor(sentences_ids), torch.LongTensor(label_ids)

        data_loader = DataLoader(TensorDataset(X, y), batch_size, shuffle)

        split[n] = data_loader
    
    return split.values()



def build_dataloader(X, y, batch_size = 1, shuffle=False):
    '''
    TensorDataset is a subclass of Dataset, 
    as long as the tensor is passed in, it is indexed by the first dimension.
    '''
    total_len = len(X)
    train_cutoff = int(0.7 * total_len)
    valid_cutoff = int(0.8 * total_len)

    rand_idx = torch.randperm(total_len)
    train_index = rand_idx[:train_cutoff]
    valid_index = rand_idx[train_cutoff:valid_cutoff]
    test_index = rand_idx[valid_cutoff:]

    X_train, y_train = torch.Tensor(X[train_index]).long(), torch.Tensor(y[train_index]).float()
    X_valid, y_valid = torch.Tensor(X[valid_index]).long(), torch.Tensor(y[valid_index]).float()
    X_test, y_test = torch.Tensor(X[test_index]).long(), torch.Tensor(y[test_index]).float()
    
    train_data = TensorDataset(X_train, y_train)
    valid_data = TensorDataset(X_valid, y_valid)
    test_data =   TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_data, batch_size, shuffle)
    valid_loader = DataLoader(valid_data, batch_size, shuffle)
    test_loader = DataLoader(test_data, batch_size, shuffle)
    
    return train_loader, valid_loader, test_loader




# def simple_clean_sentence(text):
#     '''
#     extract useful tokens from a sentence
#     '''

#     stop_words_en = stopwords.words('english')
    
#     text = text.lower()

#     text = text.translate(str.maketrans('', '', string.punctuation))
    
#     tokens = word_tokenize(text)
    
#     # remove stopping words
#     tokens = [ w for w in tokens if w not in stop_words_en ]
#     tokens = [ w for w in tokens if len(w) > 1 ]
    
#     # return tokens
#     return ' '.join(tokens)


