import torch
from torch import nn
import torch.nn.functional as F

class SentimentLSTM(nn.Module):
    def __init__(self, params):
        super().__init__()

        vocab_size, hidden_dim, embed_dim, output_dim = params

        self.vocab_size = vocab_size  
        self.hidden_dim = hidden_dim

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
        
        # fc_out: (batch_size, seq_n)
        fc_out = F.log_softmax(fc_out, dim=1)
        

