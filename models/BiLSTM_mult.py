'''
The baseline BiLSTM_CRF model with mini-batch

@reference:
  - https://github.com/zliucr/CrossNER/tree/2e7ba2a7798c961e3f29fbc51252c5a8d40224bf
'''

import torch
import torch.nn as nn
from torch.nn import init
import numpy as np
import torch.optim as optim
from models.BiLSTM_batch import CRF


class BiLSTM_CRF_Mult(nn.Module):
  def __init__(
    self, 
    vocab_size, 
    embedding_dim, 
    hidden_dim=256, 
    dropout=.5, 
    multi_domain_config=None,
    use_char_embed=False, 
    char_embedding_dim=25, 
    char_hidden_dim=64, 
    char2id=None, 
    pre_word_embedding=None,
    device=None,
  ):
    super(BiLSTM_CRF_Mult, self).__init__()

    self.device = device

    self.vocab_size = vocab_size
    self.embedding_dim = embedding_dim
    self.hidden_dim = hidden_dim
   
    # character-level
    self.use_char_embed = use_char_embed
    if use_char_embed:
      self.char_hidden_dim = char_hidden_dim
      self.char_embed = nn.Embedding(len(char2id), char_embedding_dim)
      self.char_lstm = nn.LSTM(char_embedding_dim, char_hidden_dim // 2, num_layers = 1, bidirectional=True, batch_first=True)
      self.init_lstm(self.char_lstm)

    # word-level
    self.word_embed = nn.Embedding(vocab_size, embedding_dim)
    
    # pre-trained word embedding 300-d
    self.pre_word_embedding = pre_word_embedding
    if self.pre_word_embedding is not None:
      self.word_embed.weight = nn.Parameter(torch.FloatTensor(pre_word_embedding))


    # dropout layer
    self.dropout = nn.Dropout(dropout)

    # BiLSTM layer
    if use_char_embed:
      self.lstm = nn.LSTM(embedding_dim + char_hidden_dim, hidden_dim // 2, bidirectional=True, batch_first=True)
    else:
      self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers = 1, bidirectional=True, batch_first=True)
    self.init_lstm(self.lstm)

    # well, all domains have the same label sets... should be constant = 7, too lazy to modify again
    self.fc = nn.Linear(hidden_dim, multi_domain_config['bc']['num_of_tag'])
    self.init_linear()


    # private layer - context layer, crf
    for d in multi_domain_config.keys():
      d_crf_name = d + '_crf'
      d_num_of_tag = multi_domain_config[d]['num_of_tag']
      setattr(self, d_crf_name, CRF(d_num_of_tag))



  def init_linear(self):
    init.xavier_normal_(self.fc.weight.data)
    init.normal_(self.fc.bias.data)


  def init_lstm(self, lstm):
    for param in lstm.parameters():
      if len(param.shape) >= 2:
          init.orthogonal_(param.data)
      else:
          init.normal_(param.data)

  
  def _get_domain_layer(self, domain, layer_name):
    return getattr(self, domain + '_' + layer_name)


  def forward(self, inputs, input_chars=None, word_len_per_sent=None, perm_idx=None, from_domain=None, seq_len_in_batch=None):
    '''
      inputs(after padding):
        - (batch_size, max_seq_len) 
        - each element indicates the corresponding word_id
      
      # if you use char embedding,
      input_chars(after padding):
        - (seq_len, max_word_len)
        -  each element indicates the corresponding char_id
      # [
      #   there are 3 words in this sentence
      #   the first word consist of 8 letters
      #   [ 6  9  8  4  1  11  12  10 ] # 8
      #   [ 7  3  2  5  13  7  0  0 ] # 6
      #   [ 12  5  8 14  0  0  0  0 ] # 4
      # ]
     
      @return
        emmision score: (batch_size, max_seq_len, num_of_tag)
    '''

    if self.use_char_embed:
      # (batch_size*max_seq_len, max_word_len, char_embedding_dim)
      embed_chars = self.char_embed(input_chars)

      # (sum(word_len_per_sent), char_embedding) => (8+6+4, char_embedding_dim)
      packed_char_inputs = torch.nn.utils.rnn.pack_padded_sequence(embed_chars, word_len_per_sent, batch_first=True)

      # (sum(word_len_per_sent), char_hidden_dim) => (8+6+4, char_hidden_dim)
      lstm_char_out, _ = self.char_lstm(packed_char_inputs)

      # char_outpus, (batch_size*max_seq_len, max_word_len, char_hidden_dim)
      # char_input_sizes: equal to word_len_per_word
      char_outputs, char_input_sizes = torch.nn.utils.rnn.pad_packed_sequence(lstm_char_out, batch_first=True)
      
      # concatenate the last output of BiLSTM
      concat_char_embed_sorted = torch.zeros((char_outputs.size(0), char_outputs.size(2)), dtype=torch.float)
      for i, index in enumerate(char_input_sizes):
        concat_char_embed_sorted[i] = torch.cat((
            char_outputs[ i, index - 1, :(self.char_hidden_dim // 2) ],
            char_outputs[ i, 0, (self.char_hidden_dim // 2): ],
        ))

      # the obtained concat_char_embedding is sorted in descending order by seq_len
      # however, the inputs is the original inputs, so we need map them correctly so as to concate them later
      concat_char_embed = torch.zeros_like(concat_char_embed_sorted).to(self.device)
      for i in range(concat_char_embed.size(0)):
        concat_char_embed[perm_idx[i]] = concat_char_embed_sorted[i]
      # (batch_size*max_seq_len, char_hidden_dim)
  

    if self.use_char_embed:
      # (batch_size, max_seq_len, embed_dim)
      embed_words = self.word_embed(inputs)
     
      # (batch_size, max_seq_len, char_hidden_dim)
      # for padding word, use zero vector
      concat_char_embed_expand = torch.zeros(( embed_words.shape[0], embed_words.shape[1], concat_char_embed.shape[1] ), dtype=torch.float).to(self.device)
      prev_seq_len = 0
      for i, seq_len in enumerate(seq_len_in_batch): # each sentence
        concat_char_embed_expand[i, :seq_len, :] = concat_char_embed[prev_seq_len:prev_seq_len+seq_len]
        prev_seq_len += seq_len
      # concat_char_embed_expand = concat_char_embed.reshape(embed_words.shape[0], -1, concat_char_embed.shape[1])

      total_embeds = torch.cat((embed_words, concat_char_embed_expand), -1)

      total_embeds = self.dropout(total_embeds)

      # (batch_size, max_seq_len, hidden_dim)
      # lstm_out, _ = self.lstm(total_embeds)
      seq_len_in_batch_sorted, seq_perm_idx = torch.LongTensor(seq_len_in_batch).sort(0, descending=True)
      total_embeds_sorted = total_embeds[seq_perm_idx]
      packed_word_inputs = torch.nn.utils.rnn.pack_padded_sequence(total_embeds_sorted, seq_len_in_batch_sorted, batch_first=True)
      lstm_word_out, _ = self.lstm(packed_word_inputs)
      sorted_lstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_word_out, batch_first=True)
     
      lstm_out = torch.zeros_like(sorted_lstm_out).to(self.device)
      for i in range(lstm_out.size(0)):
        lstm_out[seq_perm_idx[i]] = sorted_lstm_out[i]


    else:
      # (batch_size, max_seq_len, embed_dim)
      embed_words = self.word_embed(inputs)
    
      # (batch_size, max_seq_len, hidden_dim)
      lstm_out, _ = self.lstm(embed_words)


    lstm_out = self.dropout(lstm_out)

    # (batch_size, max_seq_len, num_of_tag)
    lstm_feats = self.fc(lstm_out)

    return lstm_feats


  def crf_decode(self, inputs, labels, seq_len_in_batch=None, from_domain=None):
    '''
    inputs: (batch_size, max_seq_len, num_of_tag)
    lengths: (batch_size, )
    '''
    d_crf = self._get_domain_layer(from_domain, 'crf')

    prediction = d_crf(inputs, labels)

    if seq_len_in_batch is not None:
      prediction = [ prediction[i, :length].data.cpu().numpy() 
        for i, length in enumerate(seq_len_in_batch) ]

    return prediction


  def crf_loss(self, inputs, true_pad_labels, from_domain=None):
    """ 
      inputs: 
        - (batch_size, max_seq_len, num_of_tag)
      y: 
        - (batch_size, max_seq_len)
    """

    d_crf = self._get_domain_layer(from_domain, 'crf')

    return d_crf.loss(inputs, true_pad_labels)

