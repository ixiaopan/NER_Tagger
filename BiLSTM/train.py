import os
import argparse
import numpy as np
import time
import torch
import torch.optim as optim

from utils import utils
from models import BiLSTM
from eval import evaluate


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='./data/toy', help="Directory containing the dataset")


def train():
  # parse domain directory
  args = parser.parse_args()
  domain_data_dir = args.data_dir

  # GPU available
  is_cuda = torch.cuda.is_available()
  device = torch.device('cuda' if is_cuda else 'cpu')

  # load parameters
  params = utils.read_json('./experiments/baseline/params.json')
  params['cuda'] = is_cuda

  # for reproducibility
  torch.manual_seed(45)

  # merge dataset params
  data_params = utils.read_json(os.path.join(domain_data_dir, 'dataset_params.json'))
  params.update(data_params)
  print('=== parameters ===')
  print(params)

  # TODO: create data loader
  train_loader, valid_loader = utils.build_ner_dataloader(
    domain_data_dir, 
    names = ['train', 'valid'], 
    batch_size=params['batch_size']
  )


  # define model
  model = BiLSTM(
    vocab_size = params['vocab_size'], 
    hidden_dim = params['hidden_dim'], 
    embedding_dim = params['word_embed_dim'], 
    tag2id = utils.read_json(os.path.join(domain_data_dir, 'tag_id.json')), 

    dropout = params['dropout'], 
    pre_word_embedding=np.load(os.path.join(domain_data_dir, 'pre_word_embedding.npy')),

    use_char_embed = True, 
    char_embedding_dim = params['char_embedding_dim'], 
    char_hidden_dim = params['char_hidden_dim'],
    char2id = utils.read_json(os.path.join(domain_data_dir, 'chars_id.json'))
  )
  if is_cuda:
    model.to(device)

  optimiser = optim.Adam(model.parameters(), lr=params['learning_rate'])
  # loss_fn = rnn.loss_fn # customized loss function
  # metrics_func = rnn.metrics_func # a dictionary containing various metrics
  print('=== model ===')
  print(model)


  # train model
  print('=== training ===')
  # best_val_acc = 0
  # for epoch in range(params['epoches']):
  #   log_per_epoch = []
   
  #   for i, (inputs, labels) in enumerate(train_loader):
  #     out = model(inputs)
  #     loss = loss_fn(out, labels)

  #     optimiser.zero_grad()
  #     loss.backward()
  #     optimiser.step()

  #     # if epoch == 0 or epoch % 10
  #     if i % (len(train_loader) // params['log_batch']) == 0:
  #       log_per_batch = { name: metrics_func[name](out, labels) for name in metrics_func.keys() }
  #       log_per_batch['loss'] = loss.item()
  #       log_per_epoch.append(log_per_batch)
      
  #   log_per_epoch = { m: round(np.mean([x[m] for x in log_per_epoch ]), 4) for m in log_per_epoch[0].keys() }
  #   print('%d/%d: train %s' % (epoch + 1, params['epoches'], log_per_epoch))

  #   # validation
  #   valid_log = evaluate(model, valid_loader, params, loss_fn, metrics_func)
  #   print('     valid ', valid_log)

  #   model.train()


if __name__ == '__main__':
  train()
