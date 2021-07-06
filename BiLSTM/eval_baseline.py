'''
Evaluate each domain using the baseline model
'''

import time
import argparse
import numpy as np
import os

from utils import utils 
from models.BiLSTM import BiLSTM_CRF


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='./data/toy', help="Directory containing the dataset")
parser.add_argument('--dataset_type', default='test', help="dataset typep")
parser.add_argument('--model_param_dir', default='./experiments/baseline', help="Directory containing model parameters")


def accuracy(outputs, pre_labels):
  '''
  outputs: (batch_n, seq_n)
  pre_labels: (batch_n, seq_n)
  '''
  
  # ignore 'O': 2，-1：padding words
  outputs = np.array(outputs)
  pre_labels = np.array(pre_labels)

  mask = outputs > 2
  outputs = outputs[mask]
  pre_labels = pre_labels[mask]

  return np.sum(outputs == pre_labels) / len(outputs) * 100


def macro_avg_precision(outputs, pre_labels):
  labels = list(set(outputs))

  precision_labels = []
  for t in labels:
    total_P_t = [ (idx, i) for idx, i in enumerate(pre_labels) if i == t ]
    TP_t = [ idx for idx, _ in total_P_t if outputs[idx] == t ]

    precision_t = 0
    if len(total_P_t):
      precision_t = len(TP_t) / len(total_P_t)

    precision_labels.append((t, precision_t, len(TP_t), len(total_P_t)))

  # print('macro', precision_labels)

  return np.mean([ p for _, p, _, _ in precision_labels ]) * 100

metrics = {
  'accuracy': accuracy,
  'macro_avg_precision': macro_avg_precision
}


def evaluate(data_dir, type, model, params, eval_dir=False):
  model.eval()
  
  eva_loss = []
  total_pre_tag = []
  total_true_tag = []
  summary_word_tag_pred = []
  
  for inputs, labels, char_inputs, word_len_in_batch, perm_idx in \
    utils.build_onto_dataloader(
      data_dir, type, 
      batch_size=params['batch_size'], 
      is_cuda=params['cuda']
  ):
    # step 1
    loss = model.neg_log_likelihood(
      inputs, labels, 
      char_inputs, word_len_in_batch, perm_idx,
      params['device'] 
    )
    eva_loss.append(loss.item())


    # step 2 prediction
    batch_ret = model( inputs, labels, char_inputs, word_len_in_batch, perm_idx, params['device'] )

    # step 3
    for (_, pre_labels) in batch_ret:
      total_pre_tag += pre_labels
    total_true_tag = [ (labels[j][ labels[j] >= 0 ]).data.tolist() for j in range(len(inputs)) ] 
    total_true_tag = [ i for s in total_true_tag for i in s ]


    # step 4
    if eval_dir:
      id_word = utils.read_json(os.path.join(data_dir, 'id_word.json'))
      id_tag = utils.read_json(os.path.join(data_dir, 'id_tag.json'))

      for i in range(inputs.shape[0]):
        mask_sent = labels[i] >= 0

        for w_id, true_t_id, pred_t_id in zip(inputs[i][mask_sent], labels[i][mask_sent], batch_ret[i][1]):
          if true_t_id.item() == -1:
            true_t_id = 'O'
          else: 
            true_t_id = id_tag[str(true_t_id.item())]

          summary_word_tag_pred.append(' '.join([ id_word[ str(w_id.item()) ], true_t_id, id_tag[str(pred_t_id)] ]))

  # end
  if eval_dir:
    utils.save_text(os.path.join(eval_dir, type + '_eval_result.json'), summary_word_tag_pred)

  summary_batch = { metric: metrics[metric](total_true_tag, total_pre_tag) for metric in metrics }
  summary_batch['loss'] = np.mean(eva_loss)
  summary_batch_str = " , ".join((type + " {}: {:.4f}").format(k, v) for k, v in summary_batch.items())
  

  return summary_batch, summary_batch_str



if __name__ == '__main__':
  args = parser.parse_args()
  data_dir, model_param_dir = args.data_dir, args.model_param_dir

  # define model
  model, params = utils.init_baseline_model(BiLSTM_CRF, data_dir, model_param_dir)
  print('=== parameters ===')
  print(params)

  print('=== model ===')
  print(model)

  # load model
  eval_dir = os.path.join(model_param_dir, data_dir.split('/')[-1])
  model = utils.load_model(os.path.join(eval_dir, 'best.pth.tar'), model)

  # save logs
  test_metrics, summary_batch_str = evaluate(data_dir, args.dataset_type, model, params, eval_dir=eval_dir)

  print(summary_batch_str)

  utils.save_json(os.path.join(eval_dir, 'eval_metric.json'), test_metrics)
  
