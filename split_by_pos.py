import os
import argparse
import numpy as np
import pandas as pd
import csv
from collections import Counter

from utils import utils

parser = argparse.ArgumentParser()
parser.add_argument('--domain', default='tc', help="domain name")
parser.add_argument('--pos', default='IN', help="which pos to study")
parser.add_argument('--breakpoint', default='0,2,4,6,10,999', help="breakpoints of pos count")


def main(domain='tc', pos_list=None, breakpoint=None):
    '''
    https://huggingface.co/flair/pos-english
    VB	Verb, base form
    VBD	Verb, past tense
    VBG	Verb, gerund or present participle
    VBN	Verb, past participle
    VBP	Verb, non-3rd person singular present
    VBZ	Verb, 3rd person singular present

    pos_list = 'IN'
    pos_list = 'VB-VBD-VBG-VBN-VBP-VBZ'
    '''

    pos_mapping = {
      'VB-VBD-VBG-VBN-VBP-VBZ': 'verb',
      'IN': 'prep'
    }

    pos_in_corpus = []
    pos_counts_by_sent = []
    with open('./data/' + domain + '/test/dataset.csv', encoding='utf8') as f:
        csvreader = csv.reader(f, delimiter=',')

        for index, row in enumerate(csvreader):
          if index == 0: # header
            continue

          sentence_flag, _, pos, _ = row
          if len(sentence_flag) > 0: # start from a new sentence   
              pos_in_sent = []  
              pos_in_corpus.append( pos_in_sent )
          pos_in_sent.append( pos )


    # counter
    for v in pos_in_corpus: # each sent
        pos_counter = Counter()
        pos_counter.update(v)
        
        total_counts = 0
        for sp in pos_list.split('-'):
            total_counts += pos_counter[sp]
        pos_counts_by_sent.append( total_counts )
                
 

    # split by breakpoints
    pos_counts_by_sent = np.array(pos_counts_by_sent)

    df_sents = utils.load_sentences(domain, 'test', col_name='sentences')
    df_labels = utils.load_sentences(domain, 'test', col_name='labels')

    sent_len = np.array([ len( sent.split() ) for sent in df_sents['sentences'] ])
    breakpoint = [ int(v) for v in breakpoint.split(',')]

    for i, v in enumerate(breakpoint[:-1]):
      next_v = breakpoint[ i + 1 ]
      sub_sentences = df_sents['sentences'][ (pos_counts_by_sent < next_v) & (pos_counts_by_sent >= v) ].tolist() # python list
      sub_labels = df_labels['labels'][ (pos_counts_by_sent < next_v) & (pos_counts_by_sent >= v) ].tolist() # python list

      assert len(sub_labels) == len(sub_sentences)
      print('[{}, {})'.format(v, next_v), '-->', len(sub_labels))

      utils.save_text(os.path.join('./data', domain, 'test_pos_'+pos_mapping[pos_list]+'_'+str(v), 'sentences.txt'), sub_sentences)
      utils.save_text(os.path.join('./data', domain, 'test_pos_'+pos_mapping[pos_list]+'_'+str(v), 'labels.txt'), sub_labels)
      utils.save_json(os.path.join('./data', domain, 'test_pos_'+pos_mapping[pos_list]+'_'+str(v), 'stats.json'), len(sub_labels))



if __name__ == '__main__':
  arg = parser.parse_args()

  print('=== split subdataset from {}/test by {}, {}==='.format(
    arg.domain, arg.pos, arg.breakpoint)
  )

  main(arg.domain, arg.pos, arg.breakpoint)
