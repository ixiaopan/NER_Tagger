'''
Evaluate each domain using the baseline model
'''

import os
import argparse
import numpy as np
from seqeval.metrics import f1_score, precision_score, recall_score, accuracy_score

from utils import utils 
from models.BiLSTM_batch import BiLSTM_CRF_Batch


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='./data/toy', help="Directory containing the dataset to be evaluated")
parser.add_argument('--sub_dataset', default='test', help="Which sub dataset to use")
parser.add_argument('--model_param_dir', default='./experiments/baseline', help="Directory containing model parameters")


def accuracy(y_true, y_pred):
  return round( accuracy_score(y_true, y_pred) * 100, 4) 


def micro_precision(y_true, y_pred):
  return round( precision_score(y_true, y_pred, average='micro', zero_division=0)  * 100, 4)


def macro_precision(y_true, y_pred):
  return round( precision_score(y_true, y_pred, average='macro', zero_division=0)* 100, 4) 


def micro_recall(y_true, y_pred):
  return round( recall_score(y_true, y_pred, average='micro', zero_division=0)* 100, 4) 


def macro_recall(y_true, y_pred):
  return round( recall_score(y_true, y_pred, average='macro', zero_division=0)* 100, 4) 


def micro_f1(y_true, y_pred):
  return round( f1_score(y_true, y_pred, average='micro', zero_division=0)* 100, 4) 


def macro_f1(y_true, y_pred):
  return round( f1_score(y_true, y_pred, average='macro', zero_division=0)* 100, 4) 


metrics = {
  'accuracy': accuracy,
  'micro_f1': micro_f1,
  'macro_f1': macro_f1,
  'micro_precision': micro_precision,
  'macro_precision': macro_precision,
  'micro_recall': micro_recall,
  'macro_recall': macro_recall
}


def evaluate_batch(data_dir, sub_dataset, model, params, eval_dir, embedding_params_dir=None):
  model.eval()
  
  id_word = utils.read_json(os.path.join(embedding_params_dir, 'id_word.json'))
  
  tag_from = embedding_params_dir
  # tag_from = data_dir
  id_tag = utils.read_json(os.path.join(tag_from, 'id_tag_batch.json'))

  total_pre_tag = [] # (all words in this batch)
  total_true_tag = []  # (all words in this batch)
  summary_word_tag_pred = []

  for inputs, labels, char_inputs, word_len_in_batch, perm_idx, seq_len_in_batch in \
    utils.build_onto_dataloader(
      data_dir, 
      sub_dataset=sub_dataset, 
      embedding_params_dir=embedding_params_dir,
      batch_size=params['batch_size'], 
      is_cuda=params['cuda'],
      enable_batch=True
    ):

    # step 1 prediction
    pre_labels = model( inputs, char_inputs, word_len_in_batch, perm_idx, seq_len_in_batch )
    pre_labels = model.crf_decode( pre_labels ) # return padded tag_ids

    # for each sentence, 
    for i, length in enumerate(seq_len_in_batch):
      # the original text without padding
      pre_label_id = pre_labels[i, :length].data.cpu().numpy().tolist()
      true_label_id = labels[i, :length].data.cpu().numpy().tolist()
      sent_words_id = inputs[i, :length].data.cpu().numpy().tolist()

      # real text
      true_label, pred_label = [], []
      for w_id, true_t_id, pred_t_id in zip(sent_words_id, true_label_id, pre_label_id):
        t = id_tag[ str(true_t_id) ]
        p = id_tag[ str(pred_t_id) ]

        true_label.append( t )
        pred_label.append( p )
        summary_word_tag_pred.append( '%15s %8s %8s' % (id_word[ str(w_id) ], t, p))

      total_true_tag.append( true_label )
      total_pre_tag.append( pred_label )


  # log...
  summary_batch = { metric: metrics[metric](total_true_tag, total_pre_tag) for metric in metrics }
  summary_batch_str = ", ".join(("{}_{}: {}").format(sub_dataset, k, v) for k, v in summary_batch.items())

  # save result for data from test/test_xx only
  if 'test' in sub_dataset:
    utils.save_text(os.path.join(eval_dir, 'eval_' + sub_dataset + '_best_result.txt'), summary_word_tag_pred)
    utils.save_text(os.path.join(eval_dir, 'eval_' + sub_dataset + '_best_metric.txt'), summary_batch_str.split(', '))
 
 
  return summary_batch, summary_batch_str, summary_word_tag_pred



if __name__ == '__main__':
  args = parser.parse_args()
  data_dir, model_param_dir = args.data_dir, args.model_param_dir

  # define model
  model, params, embedding_params_dir = utils.init_baseline_model(
    BiLSTM_CRF_Batch, 
    model_param_dir, 
    data_dir,
    enable_batch=True
  )
  print('=== parameters ===')
  print(params)
  print('=== model ===')
  print(model)


  # load best model
  transfer_method = model_param_dir.split('/')[-1]
  if transfer_method == 'baseline':
    model = utils.load_model(os.path.join(model_param_dir, data_dir.split('/')[-1], 'best.pth.tar'), model)
  else: # [pool, pool_bc, 'mult_private']
    model = utils.load_model(os.path.join(model_param_dir, transfer_method, 'best.pth.tar'), model)


  # save logs
  print('=== Score ===')
  test_metrics, summary_batch_str, _ = evaluate_batch(
    data_dir, 
    args.sub_dataset, 
    model, 
    params, 
    eval_dir = os.path.join(model_param_dir, data_dir.split('/')[-1]),
    embedding_params_dir = embedding_params_dir
  )
  print(summary_batch_str)
  