import re
import numpy as np
import os

from utils import utils

def clean_ontonotes_data(min_seq_len=2, min_word_len=1):
  print('=== cleaning dataset ===')

  # C:\\projects\\datasets\\ontonotes-release-5\\ontonotes-release-5.0\\data\\files\\data\\english\\annotations\\bc\\cctv\\00\\cctv_0000
  data = utils.read_json('./data/SafeSend-u6etyJgw6CwkFGXt/ontonotes_parsed.json')

  # metadata about the parsed data
  meta_data = {
    'total_files': 0,
    'total_sentences': 0,
    'named_entities': [],
    'bc': {
      'desc': 'broadcast conversation',
    },
    'bn': {
      'desc': 'broadcast news',
    },
    'mz': {
      'desc': 'magazine genre (Sinorama magazine)',
    },
    'nw': {
      'desc': 'newswire',
    },
    'tc': {
      'desc': 'telephone conversation(CallHome corpus)',
    },
    'wb': {
      'desc': 'web data(85K of single sentences selected to improve sense coverage)',
    },
    'pt': {
      'desc': 'pivot text',
    },
  }



  file_content_list = list(data.values())
  part_key = [
      p.replace('C:\\projects\\datasets\\ontonotes-release-5\\ontonotes-release-5.0\\data\\files\\data\\english\\annotations\\', '')
      for p in data.keys() 
  ]
  # mz\\sinorama\\10\\ectb_1002
  unique_genres = np.unique( [ g.split('\\')[0] for g in part_key ] )
  # print(domain_type_partid)
  # print(unique_genres)

  # 
  standard_splits = {}
  for d in unique_genres:
    if d not in standard_splits:
      standard_splits[d] = {}

    for t in ['valid', 'test', 'train']:
      if t not in standard_splits[d]:
        standard_splits[d][t] = []
      
      for dirpath, dirs, fnames in os.walk( os.path.join('./data/splits/', t, 'annotations', d) ):
        for f in fnames:
          if f.endswith('.gold_skel'):
            standard_splits[d][t].append(f.replace('.gold_skel', ''))
  # print(standard_splits['bc']['valid'])



  # dataset stats
  meta_data['total_files'] = len(part_key)
  meta_data['total_sentences'] = sum([ len(f) for f in file_content_list ] )

  # extract entities
  entity_type = set()
  for f in file_content_list:
    for s in f.values():
      if 'ne' in s and len(s['ne']) > 0 and 'parse_error' not in s['ne'].keys():
        entity_type.update([ t['type'] for t in s['ne'].values() ])
  meta_data['named_entities'] = list(entity_type)


  # construct data files
  for g in unique_genres:
    all_sent_count_in_genre = 0
    all_tokens_count_in_genre = 0
    all_tokens_with_ne_in_genre = 0
    
    data_splits = {
      'valid': [],
      'test': [],
      'train': []
    }
    for i, t in enumerate(part_key):
      if t.split('\\')[0] == g:
        for k in data_splits.keys():
          if t.split('\\')[-1] in standard_splits[g][k]:
            data_splits[k].append( file_content_list[i] )



    for k in data_splits.keys():
      genre_corpus = [['Sentence #', 'Word', 'POS', 'Tag']] # header
      total_sent_count_in_genre = 0
      total_tokens_count_in_genre = 0
      total_tokens_count_by_entity_in_genre = {
        'PER': 0, 
        'LOC': 0, 
        'ORG': 0
      }

      for f in data_splits[k]: # each file
        for s in f.values(): # each sentence
          total_sent_count_in_genre += 1

          tokens_in_sent, pos_in_sent = s['tokens'], s['pos']

          # skip meaningless sentence
          if len(tokens_in_sent) < min_seq_len:
            continue

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
          for i, word in enumerate(tokens_in_sent): # a token per sentence
            total_tokens_count_in_genre += 1

            word = re.sub('^/.$', '.', word)
            word = re.sub('^``$', '', word)
            word = re.sub("^''$", '', word)
            # if the word is exactly '"', it will cause error due to empty string
            word = re.sub('"', '', word) # conficts with the '"' below
            word = utils.zero_digit(word) # replace all digits with a single 0

            # skip meaningless words
            if len(word) < min_word_len:
              continue

            # Sentence: 1,Thousands,NNS,O
            # ,of,IN,O
            pos_tag = pos_in_sent[i]
            line = [
              'Sentence {}'.format(total_sent_count_in_genre) if i == 0 else '', 
              # because the default separater in '.csv' is comma, we need to quote them to avoid error
              '"' + word + '"' if ',' in word else word, 
              '"' + pos_tag + '"'  if ',' in pos_tag else pos_tag, 
              named_entities_per_sent[i]
            ]

            genre_corpus.append(line)



      total_tokens_ratio_by_entity_in_genre = {}
      total_tokens_with_ne_in_genre = sum(total_tokens_count_by_entity_in_genre.values()) # tokens that are named entities
      for t in ['PER', 'LOC', 'ORG']:
        total_tokens_ratio_by_entity_in_genre[t + '_ratio'] = '{:.2f}'.format(total_tokens_count_by_entity_in_genre[t] / total_tokens_with_ne_in_genre*100) if total_tokens_with_ne_in_genre else 0
  
  
      if k not in meta_data[g]:
        meta_data[g][k] = {}
      meta_data[g][k]['total_sentences'] = total_sent_count_in_genre
      meta_data[g][k]['total_tokens'] = total_tokens_count_in_genre
      meta_data[g][k]['total_tokens_with_ne'] = total_tokens_with_ne_in_genre
      meta_data[g][k]['total_tokens_with_ne_ratio'] = '{:.2f}'.format(total_tokens_with_ne_in_genre / total_tokens_count_in_genre*100)
      meta_data[g][k]['total_token_by_genre'] = total_tokens_count_by_entity_in_genre
      meta_data[g][k]['total_token_by_genre_ratio'] = total_tokens_ratio_by_entity_in_genre

      
      all_sent_count_in_genre += total_sent_count_in_genre
      all_tokens_count_in_genre += total_tokens_count_in_genre
      all_tokens_with_ne_in_genre += total_tokens_with_ne_in_genre


      utils.save_text('./data/{}/{}/dataset.csv'.format(g, k), genre_corpus, lambda s: ','.join(s))
      print(' - {}/{} done'.format(g, k))
    
    meta_data[g]['all_sent_count_in_genre'] = all_sent_count_in_genre
    meta_data[g]['all_tokens_count_in_genre'] = all_tokens_count_in_genre
    meta_data[g]['all_tokens_with_ne_in_genre'] = all_tokens_with_ne_in_genre
  
  utils.save_json('./data/data_summary.json', meta_data)  



def split_sent_labels():
  print('=== split sent & labels ===')

  domains = [ 'bc', 'bn', 'nw', 'mz', 'tc', 'wb' ]

  pool_data = {
    'train_sentences': [],
    'train_labels': [],
    
    'valid_sentences': [],
    'valid_labels': [],
    
    'test_sentences': [],
    'test_labels': []
  }
  
  for d in domains:
    for t in ['valid', 'test', 'train']:
      corpus, corpus_tag = utils.load_ner_data(os.path.join('./data', d, t, 'dataset.csv'))
      
      pool_data[t + '_sentences'] += corpus
      pool_data[t + '_labels'] += corpus_tag

      utils.save_text(os.path.join('./data', d, t, 'sentences.txt'), corpus, lambda s: ' '.join(s))
      utils.save_text(os.path.join('./data', d, t, 'labels.txt'), corpus_tag, lambda s: ' '.join(s))
  
    print(' - {} done'.format(d))

  for name in ['train', 'valid', 'test']:
    utils.save_text(os.path.join('./data/pool', name, 'sentences.txt'), pool_data[name + '_sentences'], lambda s: ' '.join(s))
    utils.save_text(os.path.join('./data/pool', name, 'labels.txt'), pool_data[name + '_labels'], lambda s: ' '.join(s))
  print(' - pool done')


if __name__ == '__main__':
  clean_ontonotes_data()
  split_sent_labels()
