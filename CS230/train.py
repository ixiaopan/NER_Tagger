import torch
from utils import utils

is_cuda = torch.cuda.is_available()
device = torch.device('cuda' if is_cuda else 'cpu')


def train():
  train_loader, valid_loader, test_loader = utils.build_ner_dataloader('./data/toy')



if __name__ == '__main__':
  train()

