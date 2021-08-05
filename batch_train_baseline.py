import os
import argparse
import json
import time
import numpy as np
import torch.optim as optim

from utils import utils
from models.BiLSTM_batch import BiLSTM_CRF_Batch
from batch_eval_baseline import evaluate_batch

from knockknock import teams_sender
from configparser import ConfigParser

config_parser = ConfigParser()
config_parser.read('config.ini')

parser = argparse.ArgumentParser()
parser.add_argument('--train_data_dir', default='./data/toy', help="Directory containing the dataset to be trained")
parser.add_argument('--best_metric', default='micro_f1', help="Metric used to obtain the best model")
parser.add_argument('--model_param_dir', default='./experiments/baseline', help="Directory containing model parameters")
parser.add_argument('--early_stop_num_epoch', default=5, help="Early Stop")


@teams_sender(webhook_url=config_parser['WEBHOOK']['teams'])
def train_and_evaluate(
  train_data_dir, 
  model_param_dir='./experiments/baseline', 
  best_metric='micro_f1',
  early_stop_num_epoch=5,
  model_meta = None
):
  if model_meta is None:
    # prepare model
    model, params, embedding_params_dir = utils.init_baseline_model(
      BiLSTM_CRF_Batch, 
      model_param_dir, 
      train_data_dir,
      enable_batch=True
    )
  else:
    model, params, embedding_params_dir = model_meta['model'], model_meta['params'], model_meta['embedding_params_dir']


  print('=== parameters ===')
  print(params)
  print('=== model ===')
  print(model)

  optimiser = optim.Adam(model.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])

  # train model
  print('=== training ===')
  start_time = time.time()
  training_log = [ 
    'Start training: ' + train_data_dir, 
    time.asctime( time.localtime(time.time()) ), 
    json.dumps({x: params[x] for x in params if x not in ['device']})
  ] # record epoch logs

  best_metric_score = 0
  nepoch_no_improve = 0 # early-stop
  exper_domain_dir = os.path.join(model_param_dir, train_data_dir.split('/')[-1])

  for epoch in range(params['epoches']):
    train_loader = utils.build_onto_dataloader(
      train_data_dir, 
      sub_dataset='train',
      embedding_params_dir=embedding_params_dir,
      batch_size=params['batch_size'], 
      is_cuda=params['cuda'],
      enable_batch=True
    )

    train_loss_per_epoch = []

    for inputs, labels, char_inputs, word_len_in_batch, perm_idx, _ in train_loader:
      '''
      inputs: (batch_size, max_seq_len)
      labels: (batch_size, max_seq_len)
      char_inputs: (batch_size*max_seq_len, max_word_len)
      word_len_in_batch: (batch_size*max_seq_len, )
      perm_idx: (batch_size*max_seq_len, )
      '''

      pred_y = model( inputs, char_inputs, word_len_in_batch, perm_idx, seq_len_in_batch )
      loss = model.crf_loss(pred_y, labels)

      optimiser.zero_grad()
      loss.backward()
      optimiser.step()

      train_loss_per_epoch.append(round(loss.item(), 4))


    # validation every n epoch
    if epoch == 0 or (epoch + 1) % params['log_every_epoch'] == 0:
      print('epoch %d/%d: ' % (epoch + 1, params['epoches']))

      is_best = False

      # validation
      val_metrics, val_metrics_str, summary_word_tag_pred = evaluate_batch(
        train_data_dir, 
        'valid', 
        model,
        params, 
        eval_dir = exper_domain_dir, 
        embedding_params_dir = embedding_params_dir,
      )

      if val_metrics[best_metric] >= best_metric_score:
        best_metric_score = val_metrics[best_metric]
        is_best = True
        nepoch_no_improve = 0
      else:
        nepoch_no_improve += 1


      if nepoch_no_improve >= early_stop_num_epoch:
        print("- early stopping {} epochs".format(nepoch_no_improve))
        training_log.append("- early stopping {} epochs".format(nepoch_no_improve))
        break


      utils.save_model(exper_domain_dir, {
        'epoch': epoch + 1,
        'model_dict': model.state_dict(),
        'optim_dict': optimiser.state_dict()
      }, is_best)


      if is_best:
        print('best', best_metric, best_metric_score)
        utils.save_text(os.path.join(exper_domain_dir, 'eval_valid_best_result.txt'), summary_word_tag_pred)
        utils.save_text(os.path.join(exper_domain_dir, 'eval_valid_best_metric.txt'), ('Epoch ' + str((epoch + 1)) + ', ' + val_metrics_str).split(', '))


      # log...
      epoch_log = 'train loss: %.4f, %s' % (np.mean(train_loss_per_epoch), val_metrics_str)
      training_log.append(epoch_log)
      print(epoch_log)

      # revert 
      model.train()
  

  # training done
  run_time_str = 'Training time: {}'.format(time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
  training_log.append(run_time_str)
  utils.save_text(os.path.join(exper_domain_dir, 'training_log.txt'), training_log)
  print(run_time_str)


if __name__ == '__main__':
  args = parser.parse_args()  
 
  train_and_evaluate(
    args.train_data_dir, 
    args.model_param_dir, 
    args.best_metric,
    args.early_stop_num_epoch
  )
