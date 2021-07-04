import os
import numpy as np
import torch
import torch.optim as optim

from utils import utils
from model import rnn
from evaluate import evaluate

def train():
  # GPU available
  is_cuda = torch.cuda.is_available()
  # device = torch.device('cuda' if is_cuda else 'cpu')

  # load parameters
  params = utils.read_json('./exper/params.json')
  params['cuda'] = is_cuda

  # for reproducibility
  torch.manual_seed(45)


  # merge dataset params
  data_dir = './data/toy'
  data_params = utils.read_json(os.path.join(data_dir, 'dataset_params.json'))
  params.update(data_params)
  print('=== parameters ===')
  print(params)

  # data loader TODO: 
  train_loader, valid_loader = utils.build_custom_dataloader(
    data_dir, 
    names = ['train', 'valid'], 
    batch_size=params['batch_size']
  )

  # define model
  model = rnn.LSTM(params).cuda() if is_cuda else rnn.LSTM(params)
  optimiser = optim.Adam(model.parameters(), lr=params['learning_rate'])
  loss_fn = rnn.loss_fn # customized loss function
  metrics_func = rnn.metrics_func # a dictionary containing various metrics
  print('=== model ===')
  print(model)


  # train model
  print('=== training ===')
  best_val_acc = 0
  for epoch in range(params['epoches']):
    log_per_epoch = []
   
    for i, (inputs, labels) in enumerate(train_loader):
      out = model(inputs)
      loss = loss_fn(out, labels)

      optimiser.zero_grad()
      loss.backward()
      optimiser.step()

      # if epoch == 0 or epoch % 10
      if i % (len(train_loader) // params['log_batch']) == 0:
        log_per_batch = { name: metrics_func[name](out, labels) for name in metrics_func.keys() }
        log_per_batch['loss'] = loss.item()
        log_per_epoch.append(log_per_batch)
      
    log_per_epoch = { m: round(np.mean([x[m] for x in log_per_epoch ]), 4) for m in log_per_epoch[0].keys() }
    print('%d/%d: train %s' % (epoch + 1, params['epoches'], log_per_epoch))

    # validation
    valid_log = evaluate(model, valid_loader, params, loss_fn, metrics_func)
    print('     valid ', valid_log)

    model.train()


if __name__ == '__main__':
  train()
