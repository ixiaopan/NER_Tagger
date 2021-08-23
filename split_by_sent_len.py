import os
import argparse
import numpy as np
import pandas as pd
from utils import utils
import string

parser = argparse.ArgumentParser()
parser.add_argument('--domain', default='nw', help="domain name")
parser.add_argument('--split_type', default='test', help="train/valid/test")
parser.add_argument('--sent_len_threshold', default='2,4,6,10,58,9999', help="breakpoints of sent length")


def main(domain, split_type, sent_len_threshold):
  df_sents = utils.load_sentences(domain, split_type, col_name='sentences')
  df_labels = utils.load_sentences(domain, split_type, col_name='labels')

  sent_len = []
  for sent in df_sents['sentences']:
    sent_words = sent.split()    
    if sent_words[-1] in string.punctuation:
      sent_len.append( len(sent_words) - 1 )
    else:
      sent_len.append( len(sent_words) )

  sent_len = np.array(sent_len)
  sent_len_threshold = [ int(v) for v in sent_len_threshold.split(',')]

  for i, v in enumerate(sent_len_threshold[:-1]):
    next_v = sent_len_threshold[ i + 1 ]
    sub_sentences = df_sents['sentences'][ (sent_len < next_v) & (sent_len >= v) ].tolist() # python list
    sub_labels = df_labels['labels'][ (sent_len < next_v) & (sent_len >= v) ].tolist() # python list

    assert len(sub_labels) == len(sub_sentences)
    print('[{}, {})'.format(v, next_v), '-->', len(sub_labels))

    utils.save_text(os.path.join('./data', domain, 'test_sent_'+str(v), 'sentences.txt'), sub_sentences)
    utils.save_text(os.path.join('./data', domain, 'test_sent_'+str(v), 'labels.txt'), sub_labels)
    utils.save_json(os.path.join('./data', domain, 'test_sent_'+str(v), 'stats.json'), len(sub_labels))



if __name__ == '__main__':
  arg = parser.parse_args()

  print('=== split subdataset from {}/{} by sent_len_threshold {} ==='.format(arg.domain, arg.split_type, arg.sent_len_threshold))

  main(arg.domain, arg.split_type.split(','), arg.sent_len_threshold)
