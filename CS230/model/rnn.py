import torch
from torch import nn
import torch.nn.functional as F

class LSTM(nn.Module):
    def __init__(self, params):
        super().__init__()

        vocab_size = params['vocab_size']
        embed_dim = params['embed_dim']
        hidden_dim = params['hidden_dim']
        output_dim = params['tag_size']

        self.hidden_dim  = hidden_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)


    def forward(self, inputs):
        # Embedding: (batch_size, seq_n, D)
        embed_words = self.embedding(inputs)

        # out: (batch_size, seq_n, H)
        out, h = self.lstm(embed_words)
        out = out.contiguous().view(-1, self.hidden_dim)
  
        # fc_out: (batch_size*seq_n, output_dim)
        fc_out = self.fc(out)

        fc_out = F.log_softmax(fc_out, dim=1)

        return fc_out


def loss_fn(outputs, labels):
  '''
  outputs: (batch_n*seq_n, output_dim)
  labels: (batch_n, seq_n)
  '''
  labels = torch.flatten(labels)
  
  mask = labels >= 0

  # rectify label with the value of -1, the loss will ignore it because of the mask
  labels = labels % outputs.shape[1]

  # cross-entropy
  return -torch.sum(outputs[range(outputs.shape[0]), labels]*mask) / torch.sum(mask)


def accuracy(outpus, labels):
  '''
  outputs: (batch_n*seq_n, output_dim)
  labels: (batch_n, seq_n)
  '''
  
  labels = torch.flatten(labels)

  outputs = torch.argmax(outpus, dim=1)

  # padding token has label -1, so we need to filter them
  mask = labels >= 0
  
  return (torch.sum(outputs == labels) / torch.sum(mask) * 100).item()



def precision(outpus, labels):
  pass


def reacll(outpus, labels):
  pass



def f1_score(outpus, labels):
  pass



metrics_func = {
  'accuracy': accuracy
}
