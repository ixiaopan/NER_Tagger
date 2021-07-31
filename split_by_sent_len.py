import os
import argparse
import numpy as np
import pandas as pd
from utils import utils

parser = argparse.ArgumentParser()
parser.add_argument('--domain', default='wb', help="domain name")
parser.add_argument('--split_type', default='test', help="train/valid/test")
# by IQR (total sentences(including train, test, valid) from all domains)
# [2(min), 10(Q1), 18(Q2), 29(Q3), 58(1.5IQR), 86(3IQR) ]
parser.add_argument('--sent_len_threshold', default='5,10,30,58,86', help="breakpoints of sent length")


def main(domain, split_type, sent_len_threshold):
  df_sents = utils.load_sentences(domain, split_type, col_name='sentences')
  df_labels = utils.load_sentences(domain, split_type, col_name='labels')

  sent_len = np.array([ len( sent.split() ) for sent in df_sents['sentences'] ])

  prev_len = 0
  sent_len_threshold = [ int(v) for v in sent_len_threshold.split(',')]
  if sent_len_threshold[-1] < max(sent_len):
    sent_len_threshold.append(max(sent_len))

  for v in sent_len_threshold:
    sub_sentences = df_sents['sentences'][ (sent_len<=v) & (sent_len > prev_len) ].tolist() # python list
    sub_labels = df_labels['labels'][ (sent_len<=v) & (sent_len > prev_len) ].tolist() # python list
    prev_len = v

    assert len(sub_labels) == len(sub_sentences)
    print(v, '-->', len(sub_labels))

    utils.save_text(os.path.join('./data', domain, 'test_sent_'+str(v), 'sentences.txt'), sub_sentences)
    utils.save_text(os.path.join('./data', domain, 'test_sent_'+str(v), 'labels.txt'), sub_labels)
    utils.save_json(os.path.join('./data', domain, 'test_sent_'+str(v), 'stats.json'), len(sub_labels))



if __name__ == '__main__':
  arg = parser.parse_args()

  print('=== split subdataset from {}/{} by sent_len_threshold {} ==='.format(arg.domain, arg.split_type, arg.sent_len_threshold))

  main(arg.domain, arg.split_type.split(','), arg.sent_len_threshold)
