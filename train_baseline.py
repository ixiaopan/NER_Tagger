import os
import argparse
import json
import time
import numpy as np
import torch.optim as optim

from utils import utils
from models.BiLSTM import BiLSTM_CRF
from eval_baseline import evaluate

from knockknock import teams_sender
from configparser import ConfigParser

config_parser = ConfigParser()
config_parser.read('config.ini')

parser = argparse.ArgumentParser()
parser.add_argument('--train_data_dir', default='./data/toy', help="Directory containing the dataset to be trained")
parser.add_argument('--best_metric', default='micro_f1', help="metric used to obtain the best model")
parser.add_argument('--model_param_dir', default='./experiments/baseline', help="Directory containing model parameters")
parser.add_argument('--model_weight_filepath', default=None, help="Pretrained model weights")

@teams_sender(webhook_url=config_parser['WEBHOOK']['teams'])
def train_and_evaluate(
  train_data_dir, 
  model_param_dir='./experiments/baseline', 
  best_metric='micro_f1',
  model_weight_filepath=None
):
  # baseline pool, pool_init
  transfer_method = model_param_dir.split('/')[-1] 
  if transfer_method in ['pool', 'poo_init']: # using pool
    data_params_dir = './data/pool'
  elif transfer_method == 'baseline':
    data_params_dir = train_data_dir

  # prepare model
  model, params = utils.init_baseline_model(BiLSTM_CRF, data_params_dir, model_param_dir)
  print('=== parameters ===')
  print(params)
  print('=== model ===')
  print(model)

  # whether initialise model using other pre-trained model parameters
  if model_weight_filepath is not None:
    print('Initialise model weights for', train_data_dir)
    model = utils.load_model(model_weight_filepath, model)

  optimiser = optim.Adam(model.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])

  logger = utils.Logger(params['debug'])

  # train model
  print('=== training ===')
  start_time = time.time()
  training_log = [ 
    'Start training: ' + train_data_dir, 
    time.asctime( time.localtime(time.time()) ), 
    json.dumps({x: params[x] for x in params if x not in ['device']})
  ] # record epoch logs

  best_metric_score = 0
  exper_type_dir = os.path.join(model_param_dir, train_data_dir.split('/')[-1])

  for epoch in range(params['epoches']):
    train_loss_per_epoch = []
    train_loader = utils.build_onto_dataloader(
      train_data_dir, 
      data_params_dir=data_params_dir,
      type='train', batch_size=params['batch_size'], is_cuda=params['cuda']
    )

    for inputs, labels, char_inputs, word_len_in_batch, perm_idx in train_loader:
      '''
      inputs: (batch_size, max_seq_len)
      labels: (batch_size, max_seq_len)
      char_inputs: (batch_size*max_seq_len, max_word_len)
      word_len_in_batch: word_len in batch
      '''

      logger.log('inputs shape:', inputs.shape)
      logger.log('labels shape:', labels.shape)
      logger.log('char_inputs shape:', char_inputs.shape)


      loss = model.neg_log_likelihood(
        inputs, 
        labels, 
        char_inputs, 
        word_len_in_batch, 
        perm_idx
      )

      optimiser.zero_grad()
      loss.backward()
      optimiser.step()

      train_loss_per_epoch.append(round(loss.item(), 4))


    # validation every n epoch
    if epoch == 0 or (epoch + 1) % params['log_every_epoch'] == 0:
      is_best = False
      print('epoch %d/%d: ' % (epoch + 1, params['epoches']))

      # validation
      val_metrics, val_metrics_str, summary_word_tag_pred = evaluate(train_data_dir, 'valid', model, params, eval_dir=exper_type_dir, data_params_dir=data_params_dir)

      if val_metrics[best_metric] >= best_metric_score:
        best_metric_score = val_metrics[best_metric]
        is_best = True

      utils.save_model(exper_type_dir, {
        'epoch': epoch + 1,
        'model_dict': model.state_dict(),
        'optim_dict': optimiser.state_dict()
      }, is_best)

      if is_best:
        print('best', best_metric, best_metric_score)
        utils.save_text(os.path.join(exper_type_dir, 'eval_valid_best_result.txt'), summary_word_tag_pred)
        utils.save_text(os.path.join(exper_type_dir, 'eval_valid_best_metric.txt'), ('Epoch ' + str((epoch + 1)) + ', ' + val_metrics_str).split(', '))


      # log...
      epoch_log = 'train loss: %.4f, %s' % (np.mean(train_loss_per_epoch), val_metrics_str)
      training_log.append(epoch_log)
      print(epoch_log)


      # revert 
      model.train()
  

  # training done
  utils.save_text(os.path.join(exper_type_dir, 'training_log.txt'), training_log)
  print('Training time: ', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))


if __name__ == '__main__':
  args = parser.parse_args()  
 
  train_and_evaluate(
    args.train_data_dir, 
    args.model_param_dir, 
    args.best_metric,
    args.model_weight_filepath
  )
