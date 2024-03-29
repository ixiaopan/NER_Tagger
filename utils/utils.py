"""
toolbox
"""

import os
import re
from os import path
import json
import csv
import numpy as np
import pandas as pd
from collections import Counter

import torch
from torch.utils.data import TensorDataset, DataLoader


# from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords

class Logger:
  def __init__(self, debug):
    self.debug = debug


  def log(self, *text):
    if self.debug:
      print(*text)



def save_model(filedir, state, is_best=False):
  '''
  @params:
    filedir: the directory where the file is saved
    state: the model state
  '''
  if not os.path.exists(filedir):
    os.mkdir(filedir)

  torch.save(state, os.path.join(filedir, 'last.pth.tar'))

  if is_best:
    torch.save(state, os.path.join(filedir, 'best.pth.tar'))



def load_model(filepath, model, optimiser=None):
  if not os.path.exists(filepath):
    return print('File doesn\' exist.')

  model_dict = torch.load(filepath)

  model.load_state_dict(model_dict['model_dict'])

  # print("Model's state_dict:")
  # for param_tensor in model_dict['model_dict']:
  #     print(param_tensor, "\t", model_dict['model_dict'][param_tensor].size())

  if optimiser:
    optimiser.load_state_dict(model_dict['optim_dict'])

  return model


multi_domain_config = {
  'bc': {
    'batch_size': 64,
    'num_of_tag': 7
  },
  'bn': {
    'batch_size': 48,
    'num_of_tag': 7
  },
  'mz': {
    'batch_size': 32,
    'num_of_tag': 7
  },
  'nw': {
    'batch_size': 64,
    'num_of_tag': 7
  },
  'tc': {
    'batch_size': 48,
    'num_of_tag': 7
  },
  'wb': {
    'batch_size': 64,
    'num_of_tag': 7
  }
}

def prepare_model_mult_domain(models, embedding_params_dir, model_param_dir, multi_domain_config):
  pre_word_embedding = np.load(os.path.join(embedding_params_dir, 'pre_word_embedding.npy'))
  char2id = read_json(os.path.join(embedding_params_dir, 'char_id.json'))
  # tag2id = read_json(os.path.join(embedding_params_dir, 'tag_id_batch.json'))

  params = prepare_model_params(embedding_params_dir, model_param_dir)

  model = models(
    vocab_size = params['vocab_size'], 
    hidden_dim = params['hidden_dim'], 
    embedding_dim = params['word_embed_dim'], 
    dropout = params['dropout'],
    multi_domain_config = multi_domain_config.copy(),

    pre_word_embedding = pre_word_embedding,
    use_char_embed = params['use_char_embed'], 
    char_embedding_dim = params['char_embed_dim'], 
    char_hidden_dim = params['char_hidden_dim'],
    char2id = char2id,

    device = params['device']
  )

  if params['cuda']:
    model.to(params['device'])
  
  return model, params


def prepare_model_params(embedding_params_dir, model_param_dir):
  # GPU available
  is_cuda = torch.cuda.is_available()
  device = torch.device('cuda' if is_cuda else 'cpu')

  # load model parameters
  params = read_json(os.path.join(model_param_dir, 'params.json'))
  params['cuda'] = is_cuda
  params['device'] = device

  # merge embedding params
  embed_params = read_json(os.path.join(embedding_params_dir, 'dataset_params.json'))
  params.update(embed_params)

  return params


def init_baseline_model(
  models, 
  model_param_dir, 
  train_data_dir,
  enable_batch=True
):
  '''
  initialise the baseline model

  @params:
    enable_batch: 
        - labels without __START__ and __STOP__
  '''
  # baseline pool, pool_init
  # embedding_params_dir: directory containing the vocabulary used to config NER model
  #       - individual domain: bc, mz
  #       - pool/pool_init: utilize all avaliable data from all domains
  transfer_method = model_param_dir.split('/')[-1] 

  if transfer_method == 'baseline':
    embedding_params_dir = train_data_dir
  
  elif transfer_method == 'pool_init':
    embedding_params_dir = './data/pool'

  else: # using pool
    embedding_params_dir = './data/' + transfer_method


  params = prepare_model_params(embedding_params_dir, model_param_dir)
  
  # define model
  pre_word_embedding = None
  if bool(params['use_pre_trained']):
    pre_word_embedding = np.load(os.path.join(embedding_params_dir, 'pre_word_embedding.npy'))

  char2id = read_json(os.path.join(embedding_params_dir, 'char_id.json'))


  tag_from = embedding_params_dir
  # tag_from = train_data_dir
  if enable_batch:
    tag2id = read_json(os.path.join(tag_from, 'tag_id_batch.json'))
  else:
    tag2id = read_json(os.path.join(tag_from, 'tag_id.json'))


  model = models(
    vocab_size = params['vocab_size'], 
    hidden_dim = params['hidden_dim'], 
    embedding_dim = params['word_embed_dim'], 
    tag2id = tag2id,
    dropout = params['dropout'], 

    pre_word_embedding = pre_word_embedding,
    use_char_embed = params['use_char_embed'], 
    char_embedding_dim = params['char_embed_dim'], 
    char_hidden_dim = params['char_hidden_dim'],
    char2id = char2id,

    device=params['device']
  )

  if params['cuda']:
    model.to(params['device'])

  return model, params, embedding_params_dir



def save_text(filepath, text, transform_fn=None):
  parent_directory = os.path.dirname(filepath)
  if not os.path.exists(parent_directory):
    os.makedirs(parent_directory)

  with open(filepath, 'w', encoding="utf8") as f:
    for i, line in enumerate(text):
      if transform_fn:
        line = transform_fn(line)
      
      if i < len(text) - 1:
        f.write('{}\n'.format(line))
      else:
        f.write('{}'.format(line))


def read_text(fpath, encoding="utf8"):
  with open(fpath, 'r', encoding=encoding) as f:
    return [ line.rstrip() for line in f ]


def save_json(filepath, text):
  parent_directory = os.path.dirname(filepath)
  if not os.path.exists(parent_directory):
    os.makedirs(parent_directory)

  with open(filepath, 'w') as f:
    json.dump(text, f, indent=4)



def read_json(filepath):
  with open(filepath, 'r') as f:
    data = json.load(f)

  return data


def load_sentences(domain='wb', split_type=['train', 'test', 'valid'], col_name='sentences'):
    sents = None

    for dt in ['train', 'test', 'valid']:
        if dt not in split_type:
            continue

        dt_sents = pd.read_csv('./data/' + domain + '/' + dt + '/' + col_name + '.txt', sep='\n', header=None)
        if sents is None:
            sents = dt_sents
        else:
            sents = pd.concat([sents, dt_sents], axis=0)

    sents.columns = [col_name]
    return sents

def split_train_val_test(corpus, corpus_tag, shuffle=True):
  total_len = len(corpus)
  train_cutoff = int(0.7 * total_len)
  valid_cutoff = int(0.8 * total_len)

  if shuffle:
    torch.manual_seed(45)
    rand_idx = torch.randperm(total_len)
  else:
    rand_idx = range(total_len)

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



def build_vocabulary(filename, word_counter=None, lower=True):
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
  
  i = 0
  with open(filename, 'r', encoding="utf-8") as f:
    for line in f: # for each line
      i += 1
      line = line.rstrip()
      if lower:
        word_counter.update(line.lower().split(' '))
      else:
        word_counter.update(line.split(' '))

  # return word_counter, i + 1
  return word_counter, i



def build_char(filename, char_counter=None):
  '''
  corpus: a list of sentence
  [
    'the first sentence',
    'the second sentence',
  ]
  
  return:
    - character counter
  '''
  if char_counter is None:
    char_counter = Counter()
  
  with open(filename, 'r', encoding="utf-8") as f:
    chars = ''.join([''.join(line.rstrip()) for line in f])

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



def prepare_single_sentence(sent, word2id, unk_name='_UNK'):
  '''
  convert a single sentence to a sequnce of ids
  e.g.'He went to school'
  '''
  
  return [ word2id[w] if w in word2id.keys() else word2id[unk_name] for w in sent.split(' ')]


# === NER specific ===
def load_ner_data(filename, encoding="utf8"):
  '''
  @desc
    a specific data format for NER
  
  @return
    sentence
    labels
  '''
  if not os.path.exists(filename):
    return print(filename + ' is not found.')


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
  use_char_embed=False,
  use_pre_trained=False,
  glove_word_dim=50,  # 50, 100, 200, 300
  min_tag_freq=1,
  dataset=['train', 'valid', 'test']
):
  '''
  build the vocabulary of a set of data & pre-load the pre-trained word embedding

  @param
    - data_dir: directory of each domain
    - use_pre_trained: whether to use pre-trained word embedding GloVe
    - glove_word_dim: specify the word dimension to use
  '''

  PAD_WORD = '_PAD' # fill the space for a sentence with unequal length
  UNK_WORD = '_UNK' # for word that is not present in the vocabulary
  START_TAG = '_START_' # for state transition
  STOP_TAG = '_STOP_'

  data_statistics = {
    'use_char_embed': use_char_embed,
    'use_pre_trained': use_pre_trained,
    'glove_word_dim': glove_word_dim,

    'train_sent_size': 0,
    'valid_sent_size': 0,
    'test_sent_size': 0,
    'train_label_size': 0,
    'valid_label_size': 0,
    'test_label_size': 0,

    'vocab_size': 0,
    'pad_word': PAD_WORD,
    'unk_word': UNK_WORD,

    'tag_size': 0,
    'start_tag': START_TAG,
    'stop_tag': STOP_TAG,
    'char_size': 0,
  }
  
  print('=== Build vocabulary from ', data_dir, ' ===' )
  
  # step 1
  split = {}
  for name in dataset:
    word_counter, sent_len = build_vocabulary(path.join(data_dir, name, 'sentences.txt'))
    split[name + '_word_counter'] = word_counter
    data_statistics[name + '_sent_size'] = sent_len


  tag_counter = Counter()
  for name in dataset:
    tag_counter, tag_sent_len = build_vocabulary(path.join(data_dir, name, 'labels.txt'), tag_counter, lower=False)
    data_statistics[name + '_label_size'] = tag_sent_len


  # should have the same number of lines
  for name in dataset:
    assert data_statistics[name + '_sent_size'] == data_statistics[name + '_label_size']


  # step 2
  if use_pre_trained and glove_word_dim:
    glove_path = './data/glove.6B/glove.6B.' + str(glove_word_dim) + 'd.txt' 
    if not os.path.isfile(glove_path):
      return print('File doesn\'t exist')


    glove_words = {}
    with open(glove_path, 'r', encoding="utf-8") as f:
      for line in f:
        value = line.split()
        glove_words[value[0]] = value[1:]


  # step 3
  vocab = [ w for w, c in split['train_word_counter'].items() if c >= min_word_freq ]
  # exclude uncommon words that are not present in glove either
  # for w, c in split['train_word_counter'].items():
  #   if c < min_word_freq and w in glove_words.keys():
  #     vocab.append(w)
      


  # for batch version
  tags_batch = [ t for t, c in tag_counter.items() if c >= min_tag_freq ]
  # for one sentence version
  tags = [ t for t, c in tag_counter.items() if c >= min_tag_freq ]
  tags.insert(0, START_TAG)
  tags.insert(0, STOP_TAG)

  chars = list(set([ s for w in vocab for s in w ]))
 
  chars.insert(0, PAD_WORD)
  chars.insert(0, UNK_WORD)

  if PAD_WORD not in vocab:
    vocab.insert(0, PAD_WORD)
  if UNK_WORD not in vocab:
    vocab.insert(0, UNK_WORD)

  data_statistics['vocab_size'] = len(vocab)
  data_statistics['tag_size'] = len(tags)
  data_statistics['char_size'] = len(chars)

  # for batch version
  data_statistics['tag_size_batch'] = len(tags_batch)

  word_id, inverse_word_id = map_word_id(vocab)
  tag_id, inverse_tag_id = map_word_id(tags)
  chars_id, inverse_chars_id = map_word_id(chars)
 
  # for batch version: 'O' is at the first position
  tag_id_batch = {'O': 0}
  for t in tags_batch:
    if t == 'O':
      continue
    tag_id_batch[t] = len(tag_id_batch)
  inverse_tag_id_batch = { t_id: t for (t, t_id) in tag_id_batch.items() }


  # save pre-trained word vector
  if use_pre_trained and glove_word_dim:
    pre_word_vector = np.random.rand(len(vocab), glove_word_dim)
    for w in vocab:
      if w in glove_words.keys():
        pre_word_vector[word_id[w]] = glove_words[w]

 

  # step 4 save meta data to file
  save_text(path.join(data_dir, 'vocabulary.txt'), vocab)
  save_text(path.join(data_dir, 'tags.txt'), tags)
  save_text(path.join(data_dir, 'tags_batch.txt'), tags_batch)

  save_json(path.join(data_dir, 'word_id.json'), word_id)
  save_json(path.join(data_dir, 'id_word.json'), inverse_word_id)
  
  save_json(path.join(data_dir, 'tag_id.json'), tag_id)
  save_json(path.join(data_dir, 'id_tag.json'), inverse_tag_id)

  save_json(path.join(data_dir, 'tag_id_batch.json'), tag_id_batch)
  save_json(path.join(data_dir, 'id_tag_batch.json'), inverse_tag_id_batch)

  save_json(path.join(data_dir, 'char_id.json'), chars_id)
  save_json(path.join(data_dir, 'id_char.json'), inverse_chars_id)


  if use_pre_trained and glove_word_dim:
    np.save(path.join(data_dir, 'pre_word_embedding.npy'), pre_word_vector)


  save_json(path.join(data_dir, 'dataset_params.json'), data_statistics)

  # step 5: log metadata
  print('\n'.join(['{}: {}'.format(k, v) for k, v in data_statistics.items()]))



def build_onto_dataloader(
  data_dir, 
  sub_dataset='train', 
  embedding_params_dir=None,
  batch_size = 1, 
  shuffle = True, 
  is_cuda = False, 
  enable_batch=False
):
  '''
  Refer: https://gist.github.com/HarshTrivedi/f4e7293e941b17d19058f6fb90ab0fec

  @desc
    - Customised dataloder designed for Ontonotes
  @params
    - data_dir: './data/bc'
    - sub_dataset: train, test, valid
    - embedding_params_dir: directory containing the vocabulary used to config NER model
        - individual domain: bc, mz
        - pool domain: utilize all avaliable data from all domains
  @return 
    - Generator: (batch_sent, batch_labels, batch_chars, word_len_per_sent)
  '''

  data_stats = read_json(path.join(embedding_params_dir, 'dataset_params.json'))
  word_id = read_json(path.join(embedding_params_dir, 'word_id.json'))
  id_word = read_json(path.join(embedding_params_dir, 'id_word.json'))
  char_id = read_json(path.join(embedding_params_dir, 'char_id.json'))

  tag_from = embedding_params_dir
  # tag_from = data_dir # for disjoint labels
  if enable_batch:
    tag_id = read_json(path.join(tag_from, 'tag_id_batch.json'))
  else:
    tag_id = read_json(path.join(tag_from, 'tag_id.json'))


  PAD_WORD = data_stats['pad_word']
  UNK_WORD = data_stats['unk_word']


  # Step 1
  sentences, tags = [], []
  with open(path.join(data_dir, sub_dataset, 'sentences.txt'), encoding="utf8") as f:
    for line in f: # for each line
      line = line.rstrip()
      sent = [ w.lower() for w in line.split() ] # real word string => we should do character embedding, otherwise all unknow words are encoded as '_UNK_'
      sentences.append(sent)


  with open(path.join(data_dir, sub_dataset, 'labels.txt'), encoding="utf8") as f:
    for line in f: # for each line
      tag_line = [ tag_id[t] for t in line.split() ] # lable id
      tags.append(tag_line)


  # check data integrity
  assert len(sentences) == len(tags)
  for i in range(len(sentences)):
    assert len(sentences[i]) == len(tags[i])


  # Step 2
  data_size = len(sentences)
  if shuffle:
    torch.manual_seed(45)
    rand_idx = torch.randperm(data_size)
  else:
    rand_idx = range(data_size)


  # Step 3 fetch data, how many batches given the fixed 'batch_size'
  for i in range((data_size // batch_size + (0 if data_size % batch_size == 0 else 1))):
    # define some terms
    # seq_len: the number of words in a setence
    # word_len: the number of chars in a word
    # variable_seq_len: indicate explictly that the length vary in sentences
    # variable_word_len: indicate explictly that the length vary in words
    batch_idx = rand_idx[ i*batch_size : (i+1)*batch_size ]

    # (batch_size, variable_seq_len)
    # [
    #   [9, 29, 10],
    #   [1, 32, 32, 3, 4, 3]
    # ]
    batch_sentences = [ sentences[idx] for idx in batch_idx ]
    batch_tags = [ tags[idx] for idx in batch_idx ]


    # Step 3.1 padding sentence
    # refer: https://github.com/cs230-stanford/cs230-code-examples/blob/master/pytorch/nlp/model/data_loader.py
    batch_max_len = max([len(s) for s in batch_sentences])
    seq_len_in_batch = [ len(s) for s in batch_sentences]

    # (batch_size, max_seq_len)
    batch_data = word_id[PAD_WORD] * np.ones(( len(batch_sentences), batch_max_len ), dtype=int)
    batch_labels = -1 * np.ones((len(batch_sentences), batch_max_len), dtype=int)
    # (batch_size, max_seq_len)
    # [
    #   [9, 29, 10, PAD_, PAD_, PAD_],
    #   [1, 32, 22,  3,    4,    3]
    # ]
    for j in range(len(batch_sentences)):
      cur_idx = len(batch_sentences[j])
      batch_data[j][:cur_idx] = [ word_id[w_str] if w_str in word_id else word_id[UNK_WORD] for w_str in batch_sentences[j] ]
      batch_labels[j][:cur_idx] = batch_tags[j]



    # Step 3.2 encode characters for each word in batch
    # (\sum_i^batch_size |seq_len|, variable_word_len)
    # [
    #   [6, 9, 8, 4, 1, 11, 12, 10], # 8 # each word
    #   [12, 5, 8, 14], # 4
    #   [7, 3, 2, 5, 13, 7] # 6
    #   ...
    # ]
    batch_chars = []
    for sent in batch_sentences: # each sentence
      for w_str in sent: # each word
        w_seq = []
        for s in w_str: # each charac
          if s in char_id:
            w_seq.append(char_id[s])
          else:
            w_seq.append(char_id[UNK_WORD])
        batch_chars.append(w_seq)
        

    


    # Step 3.2.1, calculate the max length of a word, [8, 4, 6]
    # (\sum_i^batch_size |seq_len|, )
    word_len_in_batch = torch.LongTensor( [ len(s) for s in batch_chars ])
    max_word_len_in_batch = word_len_in_batch.max() # 8


    # Step 3.2.2, Pad with 0s
    # (batch_size*max_seq_len, max_word_len)
    fixed_char_in_batch = char_id[PAD_WORD] * torch.ones(
      ( len(batch_chars), max_word_len_in_batch ), 
      dtype=torch.long
    )
    for idx, (seq, seqlen) in enumerate( zip(batch_chars, word_len_in_batch) ):
      fixed_char_in_batch[idx, :seqlen] = torch.LongTensor(seq)


    # Step 3.2.3, sort instances in descending order
    # [
    #   [ 6  9  8  4  1 11 12 10 ]
    #   [ 7  3  2  5 13  7  0  0 ]
    #   [ 12  5  8 14  0  0  0  0 ]
    # ]
    # both are the shape of (batch_size*max_seq_len, )
    word_len_in_batch, perm_idx = torch.sort(word_len_in_batch, dim=0, descending=True)
    fixed_char_in_batch = fixed_char_in_batch[perm_idx]



    # Step 4
    if enable_batch:
      batch_data = torch.LongTensor(batch_data)
      batch_labels = torch.LongTensor(batch_labels)
    else: # force batch_size=1
      batch_data = torch.LongTensor(batch_data.flatten())
      batch_labels = torch.LongTensor(batch_labels.flatten())

    if is_cuda:
      fixed_char_in_batch, batch_data, batch_labels = fixed_char_in_batch.cuda(), batch_data.cuda(), batch_labels.cuda()

    yield batch_data, batch_labels, fixed_char_in_batch, word_len_in_batch, perm_idx, seq_len_in_batch





def build_custom_dataloader(data_dir, names = ['train', 'valid', 'test'], batch_size = 1, shuffle = True):
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


