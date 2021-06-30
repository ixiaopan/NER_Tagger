import re
import os
import csv
import argparse
import numpy as np
from utils import utils

def clean_ontonotes_data():
  # C:\\projects\\datasets\\ontonotes-release-5\\ontonotes-release-5.0\\data\\files\\data\\english\\annotations\\bc\\cctv\\00\\cctv_0000
  data = utils.read_json('./data/SafeSend-u6etyJgw6CwkFGXt/ontonotes_parsed.json')

  # metadata about the parsed data
  meta_data = {
    'total_files': 0,
    'total_sentences': 0,
    'named_entities': [],
    'genres': {
      'bc': {
        'desc': 'broadcast conversation',
        'total_sentences': 0,
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
        'total_tokens': 0,
        'total_token_by_genre': {
          'PER': 0,
          'LOC': 0,
          'ORG': 0
        }
      },
    }
  }

  genre_each_file = np.array([
      p.replace('C:\\projects\\datasets\\ontonotes-release-5\\ontonotes-release-5.0\\data\\files\\data\\english\\annotations\\', '').split('\\')[0] 
      for p in data.keys() 
  ])
  file_content_list = list(data.values())
  
  # simple stats
  meta_data['total_files'] = len(genre_each_file)
  meta_data['total_sentences'] = sum([ len(f) for f in file_content_list ] )


  # extract entities
  entity_type = set()
  for f in file_content_list:
    for s in f.values():
      if 'ne' in s and len(s['ne']) > 0 and 'parse_error' not in s['ne'].keys():
        entity_type.update([ t['type'] for t in s['ne'].values() ])
  meta_data['named_entities'] = list(entity_type)
  
 
  # construct data files
  genres = np.unique(genre_each_file)
  for g in genres:
    genre_indexes = np.where(genre_each_file == g)[0]
    genre_file_content = [ file_content_list[i] for i in genre_indexes ]
    
    # Sentence #,Word,POS,Tag
    # Sentence: 1,Thousands,NNS,O
    # ,of,IN,O
    genre_corpus = [['Sentence #', 'Word', 'POS', 'Tag']] # header
    total_sent_count_in_genre = 0
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

        # LOC, FAC, GPE => LOC
        # PERSON => PER
        # ORG => ORG
        # Other => O
        named_entities_per_sent = ['O'] * len(tokens_in_sent)
        if 'ne' in s and len(s['ne']) > 0 and 'parse_error' not in s['ne'].keys(): # could be None
          for ne in s['ne'].values(): # loop each ner_entiry per sentence
            ne_type = ne['type']
            ne_type = 'LOC' if ne_type in ['LOC', 'FAC', 'GPE'] else ne_type
            ne_type = 'PER' if ne_type in ['PERSON'] else ne_type

            if ne_type in ['PER', 'LOC', 'ORG']:
              for i, idx in enumerate(ne['tokens']):
                total_tokens_count_by_entity_in_genre[ne_type] += 1
                named_entities_per_sent[idx] = ('B-' if i == 0 else 'I-' ) + ne_type

        # save
        total_sent_count_in_genre += 1
        for i, word in enumerate(tokens_in_sent): # a token per line
          total_tokens_count_in_genre += 1
  
          pos_tag = pos_in_sent[i]

          word = re.sub('"', '', word) # conficts with the '"' below

          line = [
            'Sentence {}'.format(total_sent_count_in_genre) if i == 0 else '', 
            '"' + word + '"' if ',' in word else word, 
            '"' + pos_tag + '"'  if ',' in pos_tag else pos_tag, 
            named_entities_per_sent[i]
          ]

          genre_corpus.append(line)


    total_tokens_ratio_by_entity_in_genre = {}
    total_tokens_with_ne_in_genre = sum(total_tokens_count_by_entity_in_genre.values()) # tokens that are named entities
    for t in ['PER', 'LOC', 'ORG']:
      total_tokens_ratio_by_entity_in_genre[t + '_ratio'] = '{:.2f}'.format(total_tokens_count_by_entity_in_genre[t] / total_tokens_with_ne_in_genre*100) if total_tokens_with_ne_in_genre else 0
  
    meta_data['genres'][g]['total_sentences'] = total_sent_count_in_genre
    meta_data['genres'][g]['total_tokens'] = total_tokens_count_in_genre
    meta_data['genres'][g]['total_tokens_with_ne'] = total_tokens_with_ne_in_genre
    meta_data['genres'][g]['total_tokens_with_ne_ratio'] = '{:.2f}'.format(total_tokens_with_ne_in_genre / total_tokens_count_in_genre*100)
    meta_data['genres'][g]['total_token_by_genre'] = total_tokens_count_by_entity_in_genre
    meta_data['genres'][g]['total_token_ratio_by_genre'] = total_tokens_ratio_by_entity_in_genre

    utils.save_text('./data/{}/dataset.csv'.format(g), genre_corpus, lambda s: ','.join(s))
    print(' - {} done'.format(g))

  utils.save_json('./data/data_summary.json', meta_data)


if __name__ == '__main__':
  # step 1
  print('Cleaning data')
  clean_ontonotes_data()

  # step 2
  print('Splitting & Save data')
  data_summary = utils.read_json('./data/data_summary.json')
  for type in data_summary['genres'].keys():
    corpus, corpus_tag = utils.load_ner_data(os.path.join('./data', type, 'dataset.csv'))
    
    corpus = np.array(corpus, dtype=object)
    corpus_tag = np.array(corpus_tag, dtype=object)

    for name, X, y in utils.split_train_val_test(corpus, corpus_tag):
      utils.save_text(os.path.join('./data', type, name, 'sentences.txt'), X, lambda s: ' '.join(s))
      utils.save_text(os.path.join('./data', type, name, 'labels.txt'), y, lambda s: ' '.join(s))

    # step 3 build vocabulary
    utils.build_ner_profile(os.path.join('./data', type))    

  print('Build ontonotes done')
