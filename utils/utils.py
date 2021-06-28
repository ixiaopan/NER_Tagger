"""
toolbox
"""
import os
import json
import string
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


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




def simple_clean_sentence(text):
    '''
    extract useful tokens from a sentence
    '''

    stop_words_en = stopwords.words('english')
    
    text = text.lower()

    text = text.translate(str.maketrans('', '', string.punctuation))
    
    tokens = word_tokenize(text)
    
    # remove stopping words
    tokens = [ w for w in tokens if w not in stop_words_en ]
    tokens = [ w for w in tokens if len(w) > 1 ]
    
    # return tokens
    return ' '.join(tokens)


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




def pad_text(text, seq_len, pad_str):
    '''
    RNN takes a fixed length of sequence
    '''

    line = None
    text_size = len(text)

    if text_size >= seq_len:
        line = text[:seq_len]
    else:
        line = [pad_str] * (seq_len - text_size) + text
    
    return line


def map_word_id(word_list):
  word_id = {}
  inverse_word_id = {}
  
  for i, word in enumerate(word_list):
      word_id[word] = i
      inverse_word_id[i] = word
  
  return word_id, inverse_word_id
