'''
Evaluate each domain using the baseline model
'''

import os
import argparse
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix as confusion_matrix_score

from utils import utils 
from models.BiLSTM import BiLSTM_CRF


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='./data/toy', help="Directory containing the dataset")
parser.add_argument('--dataset_type', default='test', help="dataset typep")
parser.add_argument('--model_param_dir', default='./experiments/baseline', help="Directory containing model parameters")

NER_TAG_ID = 2

def clean_tags(y_true, y_pred):
  '''
  y_true: (seq_n, ) python list
  y_pre: (seq_n, ) python list
  '''

  y_true = np.array(y_true)
  y_pred = np.array(y_pred)

  # ignore 2('O')，-1(padding)
  mask = y_true > NER_TAG_ID

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


def evaluate(data_dir, type, model, params, eval_dir=False):
  model.eval()
  
  id_word = utils.read_json(os.path.join(data_dir, 'id_word.json'))
  id_tag = utils.read_json(os.path.join(data_dir, 'id_tag.json'))

  total_pre_tag = []
  total_true_tag = []
  summary_word_tag_pred = []

  for inputs, labels, char_inputs, word_len_in_batch, perm_idx in \
    utils.build_onto_dataloader(data_dir, type, batch_size=params['batch_size'], is_cuda=params['cuda']):

    # step 1 prediction
    _, pre_labels = model( inputs, char_inputs, word_len_in_batch, perm_idx )
    total_pre_tag += pre_labels # python list
    total_true_tag += labels.cpu().numpy().tolist()


    # step 2
    for w_id, true_t_id, pred_t_id in zip(inputs, labels, pre_labels):
      summary_word_tag_pred.append(
        '%15s %8s %8s' % (id_word[ str(w_id.item()) ], id_tag[ str(true_t_id.item()) ], id_tag[ str(pred_t_id) ])
      )
  

  # log...
  total_true_tag, total_pre_tag = clean_tags(total_true_tag, total_pre_tag)
  summary_batch = { metric: metrics[metric](total_true_tag, total_pre_tag) for metric in metrics }
  summary_batch_str = ", ".join(("{}_{}: {}").format(type, k, v) for k, v in summary_batch.items())

  if type == 'test':
    utils.save_text(os.path.join(eval_dir, 'eval_' + type + '_best_result.txt'), summary_word_tag_pred)
    utils.save_text(os.path.join(eval_dir, 'eval_' + type + '_best_metric.txt'), summary_batch_str.split(', '))
 
  return summary_batch, summary_batch_str, summary_word_tag_pred



if __name__ == '__main__':
  args = parser.parse_args()
  data_dir, model_param_dir = args.data_dir, args.model_param_dir

  # define model
  model, params = utils.init_baseline_model(BiLSTM_CRF, data_dir, model_param_dir)
  print('=== parameters ===')
  print(params)

  print('=== model ===')
  print(model)

  exper_datatype_dir = os.path.join(model_param_dir, data_dir.split('/')[-1])

  # load model
  model = utils.load_model(os.path.join(exper_datatype_dir, 'best.pth.tar'), model)

  # save logs
  print('=== Score ===')
  test_metrics, summary_batch_str, _ = evaluate(data_dir, args.dataset_type, model, params, eval_dir=exper_datatype_dir)
  print(summary_batch_str)
  
