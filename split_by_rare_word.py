import os
import argparse
import numpy as np
import pandas as pd

import nltk

from utils import utils

parser = argparse.ArgumentParser()
parser.add_argument('--domain', default='wb', help="domain name")
parser.add_argument('--split_type', default='test', help="train/valid/test")
parser.add_argument('--rare_freq', default='1,2,4,6,999999', help="breakpoints of word freq")


def cal_word_freq(corpus):
    words = []
    for sent in corpus:
      for w in sent.split():
          words.append(w.lower())

    w_freq = nltk.FreqDist(words)

    return {w: q for w, q in w_freq.items()}, np.array(list(w_freq.keys())), np.array(list(w_freq.values()))



def load_corpus_freq(freq_threshold, split_type=['train']):
  # if not os.path.isfile('./data/rare_word_stats.json'):
  df_corpus_sents = utils.load_sentences('pool', split_type, col_name='sentences')
  _, words, freqs = cal_word_freq(df_corpus_sents['sentences'].tolist())

  rare_words_json = dict()
  for i, v in enumerate(freq_threshold[:-1]):
    next_v = freq_threshold[ i + 1 ]
    cur_word = words[ (freqs < next_v) & (freqs >= v) ].tolist()

    print('word [{}, {}) -> {}'.format(v, next_v, len(cur_word)))
    rare_words_json[ '[{}, {})'.format(v, next_v) ] = {
      'length': len(cur_word),
      'words': cur_word
    }

  return rare_words_json
    
    # utils.save_json('./data/rare_word_stats.json', rare_words_json)

  # read
  # return utils.read_json('./data/rare_word_stats.json')



def main(domain, split_type, freq_threshold):
  rare_words_mapping = load_corpus_freq(freq_threshold)

  # split subdatasets
  for t in split_type:
    sentences = dict()
    labels = dict()
    for i, v in enumerate(freq_threshold[:-1]):
      next_v = freq_threshold[ i + 1 ]

      k = '[{}, {})'.format(v, next_v)
      sentences[ k ] = []
      labels[ k ] = []


    # load data & split
    df_sents = utils.load_sentences(domain, [t], col_name='sentences')
    df_labels = utils.load_sentences(domain, [t], col_name='labels')
    for i, sent in enumerate(df_sents['sentences']):
      for k, v in rare_words_mapping.items():
        found = False
        for w in set( sent.split() ): # each unique word
          if w.lower() in v['words']:
            sentences[k].append(sent)
            labels[ k ].append( df_labels['labels'][i] )
            found = True
            break
        if found:
          break

    # save
    for v in sentences.keys():
      print('sentences', v, '-->', len(labels[v]))

      n = v.split(', ')[0][1:]
      utils.save_text(os.path.join('./data', domain, t+'_rare_'+str(n), 'sentences.txt'), sentences[v])
      utils.save_text(os.path.join('./data', domain, t+'_rare_'+str(n), 'labels.txt'), labels[v])
      utils.save_json(os.path.join('./data', domain, t+'_rare_'+str(n), 'stats.json'), len(labels[v]))
  


if __name__ == '__main__':
  arg = parser.parse_args()

  print('=== split subdataset from {}/{} by rare_word_freq {} ==='.format(arg.domain, arg.split_type, arg.rare_freq))

  main(
    arg.domain, 
    arg.split_type.split(','), 
    [ int(v) for v in arg.rare_freq.split(',') ]
  )

