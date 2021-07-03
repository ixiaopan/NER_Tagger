"""
toolbox
"""

import os
import re
from os import path
import json
import csv
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



def build_char(filename, char_counter=None):
  '''
  corpus: a list of sentence
  [
    'the first sentence',
    'the second sentence',
  ]
  
  return:
    - character counter
    - the number of character
  '''
  if char_counter is None:
    char_counter = Counter()
  
  with open(filename) as f:
    chars = ''.join([''.join(line) for line in f.read().split('\n')])
  
  char_counter.update(chars)

  return char_counter



def map_word_id(word_list):
  word_id = {}
  inverse_word_id = {}
  
  for i, word in enumerate(word_list):
      word_id[word] = i
      inverse_word_id[i] = word
  
  return word_id, inverse_word_id



def zero_digit(s):
  '''
  replace each digit with a zero
  '''
  return re.sub('\d', '0', s)



def prepare_single_sentence(sent, word2id, char2id, unk_name='_UNK'):
  '''
  sentence 'He went to school'
  '''
  
  words_id = [ word2id[w] if w in word2id.keys() else word2id[unk_name] for w in sent.split(' ')]
  
  chars_id = [ 
    [char2id[c] if c in char2id.keys() else char2id[unk_name] for c in w] 
    for w in sent.split(' ')
  ]

  return words_id, chars_id


# === NER specific ===
def load_ner_data(filename=None, encoding="utf8"):
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



def build_ner_profile(
  data_dir, 
  min_word_freq=1, 
  use_pre_trained=False,
  glove_word_dim=None, 
  augment_vocab_from_glove=False, 
  min_tag_freq=1
):
  '''
  build vocabulary for ner task,
  data_dir: directory of each domain
  '''
  
  PAD_CHAR = '_PAD' # fill the space for a sentence with unequal length
  UNK_CHAR = '_UNK' # for word that is not present in the vocabulary
  # PAD_TAG = 'O'
  START_TAG = '_START_' # for state transition
  STOP_TAG = '_STOP_'

  data_statistics = {
    'train_sentence_len': 0,
    'valid_sentence_len': 0,
    'test_sentence_len': 0,

    'train_tag_sentence_len': 0,
    'valid_tag_sentence_len': 0,
    'test_tag_sentence_len': 0,

    'vocab_size': 0,
    'pad_word': PAD_CHAR,
    'unk_word': UNK_CHAR,

    'tag_size': 0,
    'start_tag': START_TAG,
    'stop_tag': STOP_TAG,
    # 'pad_tag': PAD_TAG,

    'char_size': 0,
  }
  
  print('=== Build vocabulary from ', data_dir, ' ===' )
  
  # step 1
  split = {}
  # word_counter = Counter()
  for name in ['train', 'valid', 'test']:
    word_counter, sent_len = build_vocabulary(path.join(data_dir, name, 'sentences.txt'))
    split[name + '_word_counter'] = word_counter
    data_statistics[name + '_sentence_len'] = sent_len

  # tag_counter = Counter()
  for name in ['train', 'valid', 'test']:
    tag_counter, tag_sent_len = build_vocabulary(path.join(data_dir, name, 'labels.txt'))
    split[name + '_tag_counter'] = tag_counter
    data_statistics[name + '_tag_sentence_len'] = tag_sent_len

  # char_counter = Counter()
  for name in ['train', 'valid', 'test']:
    char_counter = build_char(path.join(data_dir, name, 'sentences.txt'))
    split[name + '_char_counter'] = char_counter
 
  # should have the same number of lines
  for name in ['train', 'valid', 'test']:
    assert data_statistics[name + '_sentence_len'] == data_statistics[name + '_tag_sentence_len']


  # step 2
  vocab = [ w for w, c in split['train_word_counter'].items() if c >= min_word_freq ]
  tags = [ t for t, c in split['train_tag_counter'].items() if c >= min_tag_freq ]
  chars = list(split['train_char_counter'].keys())
 
  if PAD_CHAR not in vocab:
    vocab.append(PAD_CHAR)

  # if PAD_TAG not in tags:
    # tags.append(PAD_TAG)

  vocab.append(UNK_CHAR)
  
  chars.append(PAD_CHAR)
  chars.append(UNK_CHAR)

  tags.append(START_TAG)
  tags.append(STOP_TAG)

  # TODO: Why??
  if use_pre_trained and glove_word_dim:
    glove_path = './data/glove.6B/glove.6B.' + str(glove_word_dim) + 'd.txt' 
    if not os.path.isfile(glove_path):
      return print('File doesn\'t exist')

    glove_words = {}
    with open(glove_path, 'r', encoding="utf-8") as f:
      for line in f:
        value = line.split()
        glove_words[value[0]] = value[1:]

    if augment_vocab_from_glove: # option 1
      split['train_word_counter'].update(glove_words.keys())
    else: # option 2
      for w in ((set(split['valid_word_counter'].keys()) - set(vocab)) | (set(split['test_word_counter'].keys()) - set(vocab))):
        if w in glove_words.keys() or w.lower() in glove_words.keys() or re.sub('\d', '0', w.lower()) in glove_words.keys():
          split['train_word_counter'].update([w])
  
    vocab = [ w for w, _ in split['train_word_counter'].items()]
    

  # step 3
  data_statistics['vocab_size'] = len(vocab)
  data_statistics['tag_size'] = len(tags)
  data_statistics['char_size'] = len(chars)

  word_id, inverse_word_id = map_word_id(vocab)
  tag_id, inverse_tag_id = map_word_id(tags)
  chars_id, inverse_chars_id = map_word_id(chars)


  # save pre-trained word vector
  if use_pre_trained and glove_word_dim:
    pre_word_vector = np.zeros((len(vocab), glove_word_dim))
    for w in vocab:
      if w in glove_words.keys():
        pre_word_vector[word_id[w]] = glove_words[w]

      elif w.lower() in glove_words.keys():
        pre_word_vector[word_id[w]] = glove_words[w.lower()]


  # step 4 save meta data to file
  save_text(path.join(data_dir, 'vocabulary.txt'), vocab)
  save_text(path.join(data_dir, 'tags.txt'), tags)
  
  save_json(path.join(data_dir, 'word_id.json'), word_id)
  save_json(path.join(data_dir, 'id_word.json'), inverse_word_id)
  
  save_json(path.join(data_dir, 'tag_id.json'), tag_id)
  save_json(path.join(data_dir, 'inverse_tag_id.json'), inverse_tag_id)

  save_json(path.join(data_dir, 'chars_id.json'), chars_id)
  save_json(path.join(data_dir, 'inverse_chars_id.json'), inverse_chars_id)


  if use_pre_trained and glove_word_dim:
    np.save(path.join(data_dir, 'pre_word_embedding.npy'), pre_word_vector)


  save_json(path.join(data_dir, 'dataset_params.json'), data_statistics)

  # step 5: log metadata
  print('\n'.join(['{}: {}'.format(k, v) for k, v in data_statistics.items()]))



def convert_ner_dataset_to_id(data_dir, name):
  '''
  convert each word to the corresponding id 
  '''
  data_stats = read_json(path.join(data_dir, 'dataset_params.json'))

  word_id = read_json(path.join(data_dir, 'word_id.json'))
  tag_id = read_json(path.join(data_dir, 'tag_id.json'))
  char_id = read_json(path.join(data_dir, 'char_id.json'))

  sentences, tags, char_sent = [], [], []
  with open(path.join(data_dir, name, 'sentences.txt')) as f:
    for line in f.read().split('\n'): # for each line
      char_sent.append([ 
        [ char_id[c if c in char_id.keys() else data_stats['unk_word']] for c in w ] 
        for w in line.split(' ') 
      ])

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
  assert len(sentences) == len(char_sent)
  for i in range(len(sentences)):
    assert len(sentences[i]) == len(tags[i])
  
  return sentences, tags, char_sent
  

def build_onto_dataloader(data_dir, batch_size = 1, shuffle = True):
  '''
    designed for Ontonotes
    data_dir: './data/bc/train'
    Generator: 
      (batch_sent, batch_chars, batch_labels)
  '''
  pass


def build_ner_custom_dataloader(data_dir, names = ['train', 'valid', 'test'], batch_size = 1, shuffle = True):
    '''
    designed for CS230
    
    data_dir: directory containing the data set
    name: ['train', 'valid', 'test']
    
    Generator: 
      (batch_data, batch_labels)
    '''

    split = {}
    # for name in ['train', 'valid', 'test']:
    #   if name in names:
    #     sentences_ids, label_ids, char_ids = convert_ner_dataset_to_id(data_dir, name)

    #     X, y = torch.LongTensor(sentences_ids), torch.LongTensor(label_ids)

    #     data_loader = DataLoader(TensorDataset(X, y), batch_size, shuffle)

    #     split[name] = data_loader
    
    # return split.values()



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


