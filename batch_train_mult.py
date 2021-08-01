import os
from shutil import copy
import torch
import json
import time
import numpy as np
import torch.optim as optim
from knockknock import teams_sender
from configparser import ConfigParser


from utils import utils
from models.BiLSTM_mult import BiLSTM_CRF_Mult
from batch_eval_mult import evaluate_batch, clean_tags,metrics

config_parser = ConfigParser()
config_parser.read('config.ini')


multi_domain_config = utils.multi_domain_config.copy()

embedding_params_dir = './data/pool'
model_param_dir = './experiments/mult_private'

best_metric = 'micro_f1'
early_stop_num_epoch = 5


@teams_sender(webhook_url=config_parser['WEBHOOK']['teams'])
def train_and_evaluate():
  '''
  all domains have the same label space(special case):
    PER, LOC, ORG
  private CRF for all domain
    linear layer
    crf layer
  shared layer:
    char embedding, char LSTM
    word embedding
    LSTM
  '''

  model, params = utils.prepare_model_mult_domain(embedding_params_dir, model_param_dir, multi_domain_config)

  print('=== parameters ===')
  print(params)
  print('=== model ===')
  print(model)

  optimiser = optim.Adam(model.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])

  # train model
  print('=== training ===')
  start_time = time.time()
  training_log = [ 
    'Start training multi_domain private crf ', 
    time.asctime( time.localtime(time.time()) ), 
    json.dumps({x: params[x] for x in params if x not in ['device']})
  ] # record epoch logs


  # calculate batches for each domain
  for d in multi_domain_config.keys():
    d_config = multi_domain_config[d]
    with open(os.path.join('./data', d, 'train/labels.txt'), encoding="utf8") as f:
      d_config['lines'] = len(f.readlines())
    d_config['batches'] = (d_config['lines'] // d_config['batch_size'] + (0 if d_config['lines'] % d_config['batch_size'] == 0 else 1))

  min_batches = np.min( [ v['batches'] for v in multi_domain_config.values() ])


  best_metric_score = 0
  nepoch_no_improve = 0 # early-stop, avg validation loss from each domain
  exper_domain_dir = os.path.join(model_param_dir, model_param_dir.split('/')[-1])
  for epoch in range(params['epoches']):
    train_loss_per_epoch = []

    # initialise generator
    for d in multi_domain_config.keys():
      d_train_data_dir = './data/' + d
      d_config = multi_domain_config[d]

      # https://medium.com/analytics-vidhya/a-primer-on-multi-task-learning-part-2-a0f00796d0e5
      # Sample instances in proportion to their dataset size.
      d_train_loader = utils.build_onto_dataloader(
        d_train_data_dir, 
        split_type='train',
        embedding_params_dir=embedding_params_dir,
        batch_size=d_config['batch_size'], 
        is_cuda=params['cuda'],
        enable_batch=True
      )

      d_config['train_loader'] = d_train_loader


    # start training
    for it in range(min_batches): # each batch
      loss_all_domains_iter = []
      
      for d in multi_domain_config.keys():
        d_config = multi_domain_config[d]

        inputs, labels, char_inputs, word_len_in_batch, perm_idx, _ = next(d_config['train_loader'])

        '''
        inputs: (batch_size, max_seq_len)
        labels: (batch_size, max_seq_len)
        char_inputs: (batch_size*max_seq_len, max_word_len)
        word_len_in_batch: (batch_size*max_seq_len, )
        perm_idx: (batch_size*max_seq_len, )
        '''
        pred_y = model( inputs, char_inputs, word_len_in_batch, perm_idx, from_domain=d )
        loss = model.crf_loss(pred_y, labels, from_domain=d)
        
        loss_all_domains_iter.append(loss)

      sum_loss = np.sum(loss_all_domains_iter)

      optimiser.zero_grad()
      sum_loss.backward()
      optimiser.step()

      train_loss_per_epoch.append(round(sum_loss.item(), 4))


    # validation every n epoch
    if epoch == 0 or (epoch + 1) % params['log_every_epoch'] == 0:
      print('epoch %d/%d: ' % (epoch + 1, params['epoches']))
    
      is_best = False

      total_pre_tag = []
      total_true_tag = []
      for d in multi_domain_config.keys():
        # val_metrics, val_metrics_str, summary_word_tag_pred = evaluate_batch(
        d_pre_tag, d_true_tag = evaluate_batch(
          './data/' + d, 
          'valid', 
          model,
          params,
          eval_dir = exper_domain_dir, 
          embedding_params_dir = embedding_params_dir,
          from_domain=d
        )
        total_pre_tag += d_pre_tag
        total_true_tag += d_true_tag
      
      total_true_tag, total_pre_tag = clean_tags(total_true_tag, total_pre_tag)
      val_metrics = { metric: metrics[metric](total_true_tag, total_pre_tag) for metric in metrics }
      val_metrics_str = ", ".join(("mult_valid_{}: {}").format(k, v) for k, v in summary_batch.items())

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
        utils.save_text(os.path.join(exper_domain_dir, 'eval_valid_best_metric.txt'), ('Epoch ' + str((epoch + 1)) + ', ' + val_metrics_str).split(', '))


      # log...
      epoch_log = 'train loss: %.4f, %s' % (np.mean(train_loss_per_epoch), val_metrics_str)
      training_log.append(epoch_log)
      print(epoch_log)

      # revert 
      model.train()


  # training done
  utils.save_text(os.path.join(exper_domain_dir, 'training_log.txt'), training_log)
  print('Training time: ', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))



if __name__ == '__main__':
  train_and_evaluate()
