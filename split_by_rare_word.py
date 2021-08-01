import os
import argparse
import numpy as np
import pandas as pd

import nltk
# import string
# from nltk.corpus import stopwords

from utils import utils

parser = argparse.ArgumentParser()
parser.add_argument('--domain', default='wb', help="domain name")
parser.add_argument('--split_type', default='test', help="train/valid/test")
parser.add_argument('--freq_threshold', default='7,12,25,50,100', help="breakpoints of word freq")

# stop = stopwords.words('english')
def cal_word_freq(corpus):
    words = []
    for sent in corpus:
      for w in sent.split():
          words.append(w)

    print('total words ', len(words))
    
    w_freq = nltk.FreqDist(words)

    return {w: q for w, q in w_freq.items()}, np.array(list(w_freq.keys())), np.array(list(w_freq.values()))


def load_corpus_freq(freq_threshold, split_type=['train', 'valid', 'test']):
  if not os.path.isfile('./data/rare_word_stats.json'):
    df_corpus_sents = utils.load_sentences('pool', split_type, col_name='sentences')
    _, words, freqs = cal_word_freq(df_corpus_sents['sentences'].tolist())

    rare_words_json = dict()
    prev_freq = 0
    for v in freq_threshold:
      cur_word = words[ (freqs <= v) & (freqs > prev_freq) ].tolist()    

      print('( {}, {}] -> {}'.format(prev_freq, v, len(cur_word)))
      rare_words_json[ '( {}, {}]'.format(prev_freq, v) ] = {
        'threshold': v,
        'length': len(cur_word),
        'words': cur_word
      }

      prev_freq = v
    
    utils.save_json('./data/rare_word_stats.json', rare_words_json)

 
  # read
  rare_words_json = utils.read_json('./data/rare_word_stats.json')
  rare_words_mapping = dict()
  for v in rare_words_json.values():
    for w in v['words']:
      rare_words_mapping[w] = v['threshold']

  return rare_words_mapping


def main(domain, split_type, freq_threshold):
  rare_words_mapping = load_corpus_freq(freq_threshold)

  # split subdatasets
  for t in split_type:
    sentences = dict()
    labels = dict()
    for v in freq_threshold:
      sentences[v] = []
      labels[v] = []
    sentences['common'] = []
    labels['common'] = []

    df_sents = utils.load_sentences(domain, [t], col_name='sentences')
    df_labels = utils.load_sentences(domain, [t], col_name='labels')

    for i, sent in enumerate(df_sents['sentences']):
      counter = dict()
      for w in sent.split(): # each word
        if w in rare_words_mapping:  
          if rare_words_mapping[w] not in counter:
            counter[rare_words_mapping[w]] = 0
          counter[rare_words_mapping[w]] += 1

      # select the maximum for each sentence
      if len(counter) == 0:
        max_key = 'common'
      else:
        max_key = max(counter, key=counter.get)
      sentences[max_key].append(sent)
      labels[max_key].append(df_labels['labels'][i])
      assert len(sent.split()) == len(df_labels['labels'][i].split())


    for v in sentences.keys():
      print(v, '-->', len(labels[v]))
      utils.save_text(os.path.join('./data', domain, t+'_rare_'+str(v), 'sentences.txt'), sentences[v])
      utils.save_text(os.path.join('./data', domain, t+'_rare_'+str(v), 'labels.txt'), labels[v])
      utils.save_json(os.path.join('./data', domain, t+'_rare_'+str(v), 'stats.json'), len(labels[v]))
  


if __name__ == '__main__':
  arg = parser.parse_args()
  print('=== split subdataset from {}/{} by rare_word_freq {} ==='.format(arg.domain, arg.split_type, arg.freq_threshold))

  main(
    arg.domain, 
    arg.split_type.split(','), 
    [ int(v) for v in arg.freq_threshold.split(',') ]
  )

