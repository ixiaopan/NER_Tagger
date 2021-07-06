import os
import argparse
import numpy as np
import time
import torch
import torch.optim as optim

from utils import utils
from models.BiLSTM import BiLSTM_CRF
from eval_baseline import evaluate

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='./data/toy', help="Directory containing the dataset")
parser.add_argument('--best_metric', default='accuracy', help="metric used to obtain the best model")
parser.add_argument('--model_param_dir', default='./experiments/baseline', help="Directory containing model parameters")


def train_and_evaluate(
  domain_data_dir, 
  model_param_dir='./experiments/baseline', 
  best_metric='accuracy'
):

  # prepare model
  model, params = utils.init_baseline_model(BiLSTM_CRF, domain_data_dir, model_param_dir)
  print('=== parameters ===')
  print(params)
  print('=== model ===')
  print(model)

  optimiser = optim.Adam(model.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])

  logger = utils.Logger(params['debug'])


  # train model
  print('=== training ===')
  start_time = time.time()
  training_log = [ time.asctime( time.localtime(time.time()) ) ] # record epoch logs
  
  best_metric_score = 0

  for epoch in range(params['epoches']):
    train_loss_per_epoch = []
    is_best = False

    train_loader = utils.build_onto_dataloader(
      domain_data_dir, 
      'train', 
      batch_size=params['batch_size'], 
      is_cuda=params['cuda']
    )

    for i, (inputs, labels, char_inputs, word_len_in_batch, perm_idx) in enumerate(train_loader):
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
        perm_idx,
        params['device']
      )

      optimiser.zero_grad()
      loss.backward()
      optimiser.step()

      if i % params['log_every_sent'] == 0:
        train_loss_per_epoch.append(round(loss.item(), 4))


    if epoch == 0 or (epoch + 1) % params['log_every_epoch'] == 0:
      print('epoch %d/%d: ' % (epoch + 1, params['epoches']))

      # validation
      val_metrics, val_metrics_str = evaluate(
        domain_data_dir, 
        'valid', 
        model, 
        params,
        eval_dir = os.path.join(model_param_dir, domain_data_dir.split('/')[-1])
      )

      
      if val_metrics[best_metric] >= best_metric_score:
        best_metric_score = val_metrics[best_metric]
        is_best = True

      utils.save_model(os.path.join(model_param_dir, domain_data_dir.split('/')[-1]), {
        'epoch': epoch + 1,
        'model_dict': model.state_dict(),
        'optim_dict': optimiser.state_dict()
      }, is_best)



      # log...
      epoch_log = 'train loss: %.4f, %s' % (np.mean(train_loss_per_epoch), val_metrics_str)
      training_log.append(epoch_log)
      print(epoch_log)


      # revert 
      model.train()
  



  # training done
  utils.save_text(os.path.join(model_param_dir, domain_data_dir.split('/')[-1], 'training_log.txt'), training_log)
  print('Training time: ', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))






if __name__ == '__main__':
  args = parser.parse_args()
  
  train_and_evaluate(args.data_dir, args.model_param_dir, args.best_metric)

