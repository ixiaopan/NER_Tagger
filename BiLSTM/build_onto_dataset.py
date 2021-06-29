import os
import csv
import argparse
import numpy as np
from utils import utils


def load_ontonotes_data():
  # C:\\projects\\datasets\\ontonotes-release-5\\ontonotes-release-5.0\\data\\files\\data\\english\\annotations\\bc\\cctv\\00\\cctv_0000
  data = utils.read_json('./data/SafeSend-u6etyJgw6CwkFGXt/ontonotes_parsed.json')

  # metadata about the parsed data
  meta_data = {
    'total_files': 0,
    'total_sentences': 0,
    'max_sentence_len': 0,
    'min_sentence_len': 0,
    'avg_sentence_len': 0,

    'named_entities': [],
    'genres': {
      'bc': {
        'desc': 'broadcast conversation',
        
        'total_sentences': 0,
        'max_sentence_len': 0,
        'min_sentence_len': 0,
        'avg_sentence_len': 0,
        
        'total_tokens': 0,
        'total_token_by_genre': {
          'PER': 0,
          'LOC': 0,
          'ORG': 0
        }
      },
      'bn': {
        'desc': 'broadcast news',
        'total_sentences': 0,
        'max_sentence_len': 0,
        'min_sentence_len': 0,
        'avg_sentence_len': 0,
        'total_tokens': 0,
        'total_token_by_genre': {
          'PER': 0,
          'LOC': 0,
          'ORG': 0
        }
      },
      'mz': {
        'desc': 'magazine genre (Sinorama magazine)',
        'total_sentences': 0,
        'max_sentence_len': 0,
        'min_sentence_len': 0,
        'avg_sentence_len': 0,
        'total_tokens': 0,
        'total_token_by_genre': {
          'PER': 0,
          'LOC': 0,
          'ORG': 0
        }
      },
      'nw': {
        'desc': 'newswire',
        'total_sentences': 0,
        'max_sentence_len': 0,
        'min_sentence_len': 0,
        'avg_sentence_len': 0,
        'total_tokens': 0,
        'total_token_by_genre': {
          'PER': 0,
          'LOC': 0,
          'ORG': 0
        }
      },
      'tc': {
        'desc': 'telephone conversation(CallHome corpus)',
        'total_sentences': 0,
        'max_sentence_len': 0,
        'min_sentence_len': 0,
        'avg_sentence_len': 0,
        'total_tokens': 0,
        'total_token_by_genre': {
          'PER': 0,
          'LOC': 0,
          'ORG': 0
        }
      },
      'wb': {
        'desc': 'web data(85K of single sentences selected to improve sense coverage)',
        'total_sentences': 0,
        'max_sentence_len': 0,
        'min_sentence_len': 0,
        'avg_sentence_len': 0,
        'total_tokens': 0,
        'total_token_by_genre': {
          'PER': 0,
          'LOC': 0,
          'ORG': 0
        }
      },
      'pt': {
        'desc': 'pivot text',
        'total_sentences': 0,
        'max_sentence_len': 0,
        'min_sentence_len': 0,
        'avg_sentence_len': 0,
        'total_tokens': 0,
        'total_token_by_genre': {
          'PER': 0,
          'LOC': 0,
          'ORG': 0
        }
      },
    }
  }

  genre_each_file = np.array([ p.replace('C:\\projects\\datasets\\ontonotes-release-5\\ontonotes-release-5.0\\data\\files\\data\\english\\annotations\\', '').split('\\')[0] for p in data.keys() ])
  file_content_list = list(data.values())
  
  # simple stats
  total_files = len(genre_each_file)
  total_sentence_per_file = [ len(f) for f in file_content_list ] 

  meta_data['total_files'] = total_files
  meta_data['total_sentences'] = np.sum(total_sentence_per_file)
  meta_data['max_sentence_len'] = np.max(total_sentence_per_file)
  meta_data['min_sentence_len'] = np.min(total_sentence_per_file)
  meta_data['ave_sentence_len'] = np.mean(total_sentence_per_file)

  # extract genres
  genres = np.unique(genre_each_file)

  # extract entities
  entity_type = set()
  for f in file_content_list:
    for s in f.values():
      if 'ne' in s and len(s['ne']) > 0 and 'parse_error' not in s['ne'].keys():
        entity_type.update([ t['type'] for t in s['ne'].values() ])
  meta_data['named_entities'] = entity_type
  
  print(meta_data)

  # construct data files
  for g in genres:
    genre_indexes = np.where(genre_each_file == g)[0]
    genre_file_content = [ file_content_list[i] for i in genre_indexes ]
    
    # Sentence #,Word,POS,Tag
    # Sentence: 1,Thousands,NNS,O
    # ,of,IN,O
    genre_corpus = [['Sentence #', 'Word', 'POS', 'Tag']] # header
    total_sent_count_in_genre = 0
    total_sentence_per_genre_file = [ len(f) for f in genre_file_content ] 
    
    total_tokens_count_in_genre = 0
    total_tokens_count_by_entity_in_genre = {
      'PER': 0,
      'LOC': 0,
      'ORG': 0
    }
    for f in genre_file_content: # each file
      for s in f.values(): # each sentence
        tokens_in_sent, pos_in_sent = s['tokens'], s['pos']
        
        # check data integrity
        assert len(tokens_in_sent) == len(pos_in_sent)
        

        named_entities_per_sent = ['O'] * len(tokens_in_sent)
        if 'ne' in s and len(s['ne']) > 0 and 'parse_error' not in s['ne'].keys(): # could be None
          # LOC, FAC, GPE => LOC
          # PERSON => PER
          # ORG => ORG
          # Other => O
          named_entities_in_sent = s['ne'] 


        total_sent_count_in_genre += 1
        for i, word in enumerate(tokens_in_sent): # a token per line
          total_tokens_count_in_genre += 1
          
          pos_tag = pos_in_sent[i]

          line = [ 
            'Sentence {}'.format(total_sent_count_in_genre) if i == 0 else '', 
          
            '"' + word + '"' if word == ',' else word,
          
            '"' + pos_tag + '"'  if pos_tag == ',' else pos_tag,

            '#'
            # named_entities_in_sent[i]
          ]

          genre_corpus.append(line)

    utils.save_text('./data/{}/dataset.txt'.format(g), genre_corpus, lambda s: ','.join(s))


if __name__ == '__main__':
  load_ontonotes_data()
