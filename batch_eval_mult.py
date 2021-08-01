'''
Evaluate each domain using the baseline model
'''

import os
import argparse
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix as confusion_matrix_score

from utils import utils 
from models.BiLSTM_mult import BiLSTM_CRF_Mult

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='./data/toy', help="dataset to be tested")
parser.add_argument('--split_type', default='test', help="subdataset type")
parser.add_argument('--model_param_dir', default='./experiments/mult_private', help="Directory containing model parameters")


def clean_tags(y_true, y_pred):
  '''
  y_true: (seq_n, ) python list
  y_pre: (seq_n, ) python list
  '''

  y_true = np.array(y_true)
  y_pred = np.array(y_pred)

  # ignore 2('O')
  mask = y_true > 0

  # effective NER tags
  y_true = y_true[mask]
  y_pred = y_pred[mask]

  return y_true, y_pred


def accuracy(y_true, y_pred):
  return round(accuracy_score(y_true, y_pred)*100, 4)


def confusion_matrix(y_true, y_pred):
  return confusion_matrix_score(y_true, y_pred)


def micro_precision(y_true, y_pred):
  return round(precision_score(y_true, y_pred, average='micro', zero_division=0)*100, 4)


def macro_precision(y_true, y_pred):
  return round(precision_score(y_true, y_pred, average='macro', zero_division=0)*100, 4)

def micro_recall(y_true, y_pred):
  return round(recall_score(y_true, y_pred, average='micro', zero_division=0)*100, 4)

def macro_recall(y_true, y_pred):
  return round(recall_score(y_true, y_pred, average='macro', zero_division=0)*100, 4)

def micro_f1(y_true, y_pred):
  return round(f1_score(y_true, y_pred, average='micro', zero_division=0)*100, 4)

def macro_f1(y_true, y_pred):
  return round(f1_score(y_true, y_pred, average='macro', zero_division=0)*100, 4)


metrics = {
  'accuracy': accuracy,
  'micro_f1': micro_f1,
  'macro_f1': macro_f1,
  'micro_precision': micro_precision,
  'macro_precision': macro_precision,
  'micro_recall': micro_recall,
  'macro_recall': macro_recall,
  # 'confusion_matrix': confusion_matrix
}


def evaluate_batch(
  data_dir, 
  split_type, 
  model, 
  params, 
  eval_dir,
  embedding_params_dir=None,
  from_domain=None
):
  model.eval()
  
  id_word = utils.read_json(os.path.join(embedding_params_dir, 'id_word.json'))
  
  tag_from = embedding_params_dir
  # tag_from = data_dir
  id_tag = utils.read_json(os.path.join(tag_from, 'id_tag_batch.json'))

  total_pre_tag = [] # (all words in this batch)
  total_true_tag = []  # (all words in this batch)
  total_words = []
  summary_word_tag_pred = []

  for inputs, labels, char_inputs, word_len_in_batch, perm_idx, seq_len_in_batch in \
    utils.build_onto_dataloader(
      data_dir, 
      split_type=split_type, 
      embedding_params_dir=embedding_params_dir,
      batch_size=params['batch_size'], 
      is_cuda=params['cuda'],
      enable_batch=True
    ):


    # step 1 prediction
    pre_labels = model( inputs, char_inputs, word_len_in_batch, perm_idx, from_domain=from_domain )
    pre_labels = model.crf_decode( pre_labels, from_domain=from_domain ) # return padded tag_ids


    # for each sentence, the original text without padding
    for i, length in enumerate(seq_len_in_batch):
      total_pre_tag += pre_labels[i, :length].data.cpu().numpy().tolist()
      total_true_tag += labels[i, :length].data.cpu().numpy().tolist()
      total_words += inputs[i, :length].data.cpu().numpy().tolist()


    # step 2
    if 'test' in split_type:
      for w_id, true_t_id, pred_t_id in zip(total_words, total_true_tag, total_pre_tag):
        summary_word_tag_pred.append(
          '%15s %8s %8s' % (id_word[ str(w_id) ], id_tag[ str(true_t_id) ], id_tag[ str(pred_t_id) ])
        )
      

  if 'test' in split_type:
    # log...
    total_true_tag, total_pre_tag = clean_tags(total_true_tag, total_pre_tag)
    summary_batch = { metric: metrics[metric](total_true_tag, total_pre_tag) for metric in metrics }
    summary_batch_str = ", ".join(("{}_{}: {}").format(split_type, k, v) for k, v in summary_batch.items())

    utils.save_text(os.path.join(eval_dir, 'eval_' + split_type + '_best_result.txt'), summary_word_tag_pred)
    utils.save_text(os.path.join(eval_dir, 'eval_' + split_type + '_best_metric.txt'), summary_batch_str.split(', '))
 
    return summary_batch, summary_batch_str, summary_word_tag_pred


  else:
    return total_pre_tag, total_true_tag




if __name__ == '__main__':
  domain_type = data_dir.split('/')[-1]
  embedding_params_dir = './data/pool'
  multi_domain_config = utils.multi_domain_config.copy()

  args = parser.parse_args()
  data_dir, model_param_dir = args.data_dir, args.model_param_dir

  # init model
  model, params = utils.prepare_model_mult_domain(BiLSTM_CRF_Mult, embedding_params_dir, model_param_dir, multi_domain_config)

  print('=== parameters ===')
  print(params)
  print('=== model ===')
  print(model)


  # load best model
  model = utils.load_model(os.path.join(model_param_dir, model_param_dir.split('/')[-1], 'best.pth.tar'), model)


  # save logs
  print('=== Score ===')
  
  test_metrics, summary_batch_str, _ = evaluate_batch(
    data_dir, 
    args.split_type,
    model, 
    params, 
    eval_dir = os.path.join(model_param_dir, data_dir.split('/')[-1]),
    embedding_params_dir = embedding_params_dir,
    from_domain=domain_type
  )
  print(summary_batch_str)
  