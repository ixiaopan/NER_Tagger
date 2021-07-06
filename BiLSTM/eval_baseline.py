'''
Evaluate each domain using the baseline model
'''

import torch
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

  return np.sum([i == j for i, j in zip(outputs, pre_labels)]) / len(outputs) * 100


metrics = {
  'accuracy': accuracy
}


def evaluate(data_dir, type, model, params):
  model.eval()
  
  eva_loss = []
  pre_tag = []
  true_tag = []
  
  for inputs, labels, char_inputs, word_len_in_batch, perm_idx in \
    utils.build_onto_dataloader(
      data_dir, type, 
      batch_size=params['batch_size'], 
      is_cuda=params['cuda']
  ):
    
    loss = model.neg_log_likelihood(
      inputs, labels, 
      char_inputs, word_len_in_batch, perm_idx,
      params['device'] 
    )
    eva_loss.append(loss.item())

    batch_ret = model(inputs, char_inputs, word_len_in_batch, perm_idx, params['device'] )
    for (_, pre_labels) in batch_ret:
      pre_tag += pre_labels

    true_tag += labels.view(-1).data.cpu().numpy().tolist()    

  # end
  summary_batch = { metric: metrics[metric](true_tag, pre_tag) for metric in metrics }
  summary_batch['loss'] = np.mean(eva_loss)

  return summary_batch



if __name__ == '__main__':
  args = parser.parse_args()
  data_dir, model_param_dir = args.data_dir, args.model_param_dir

  # define model
  model, params = utils.init_baseline_model(BiLSTM_CRF, data_dir, model_param_dir)
  print('=== parameters ===')
  print(params)

  print('=== model ===')
  print(model)

  eval_dir = os.path.join(model_param_dir, data_dir.split('/')[-1])
  model = utils.load_model(os.path.join(eval_dir, 'best.pth.tar'), model)

  test_metrics = evaluate(data_dir, args.dataset_type, model, params)
  utils.save_json(os.path.join(eval_dir, 'eval_metric.json'), test_metrics)
