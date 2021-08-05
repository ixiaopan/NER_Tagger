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

class CRF(nn.Module):
  def __init__(self, num_of_tag):
    super(CRF, self).__init__()
    
    self.num_of_tag = num_of_tag
    # T_{ij} transfer from i to j,
    self.transitionMatrix = nn.Parameter(torch.randn(num_of_tag, num_of_tag))
    self.start_transitions = nn.Parameter(torch.randn(num_of_tag))
    self.stop_transitions = nn.Parameter(torch.randn(num_of_tag))

  
  def _log_sum_exp(self, vec, dim):
    '''
    vec: (m, n)
    max_val: (n, ) => (1, n) if dim=0
    
    @return
      (n,)
    '''
    max_val, _ = vec.max(dim)

    return max_val + (vec - max_val.unsqueeze(dim)).exp().sum(dim).log()


  def loss(self, feats, tags):
    """
    feats: 
      - (batch_size, max_seq_len, num_of_tag)
    tags: 
      - (batch_size, max_seq_len) Should be between 0 and num_tags
    """
    # Shape checks
    assert len(feats.shape) == 3
    assert len(tags.shape) == 2
    assert feats.shape[:2] == tags.shape

    sequence_score = self._sequence_score(feats, tags)

    partition_function = self._partition_function(feats)

    log_probability = sequence_score - partition_function

    # Average across batch
    return -log_probability.mean()

  
  def _sequence_score(self, feats, tags):
      """
      feats: 
        - (batch_size, max_seq_len, num_of_tag)
        [
          [
                   O   B   I
            [ w10 0.1 0.2 0.7 ]
            [ w11 0.2 0.4 0.4 ]
            ...
          ]

          [
                  O   B   I
            [ w20 0.1 0.2 0.7 ]
            [ w21 0.2 0.4 0.4 ]
            ...
          ]
        ]
      tags: 
        - (batch_size, max_seq_len) Should be between 0 and num_tags
          [ 
            [ w10=>B, w11=>I, ...]  

            [ w20=>B, w21=>O, ...] 
            ...
          ]

      @return
        - (batch_size, )
      """
      batch_size = feats.shape[0]

      # first, select the right tag for each word
      #   (batch_size, max_seq_len, num_of_tag) gather (batch_size, max_seq_len, 1)
    
      # then, returns a tensor with all the dimensions of input of size 1 removed.
      #   (batch_size, max_seq_len, 1) =>  (batch_size, max_seq_len) 
    
      # finnaly, we get emission score for each sentence in this batch
      #   (batch_size, )

      feat_score = feats.gather(2, tags.unsqueeze(-1)).squeeze(-1).sum(dim=-1)
      # print(feat_score.size())


      # transition matrix
      # [
      #   [ [w11, w12], [w12, w13], [w13, w14], ... [w_1n-1, w1n] ],
      #   [ [w21, w22], [w22, w23], [w23, w24], ... [w_2n-1, w2n] ],
      # ]
      tags_pairs = tags.unfold(1, 2, 1)

      # Use advanced indexing to pull out required transition scores
      # (2, batch_size, max_seq_len - 1)
      # [
      #   [
      #     [w11, w12, w13 ... w_1n-1]
      #     [w21, w22, w23 ... w_2n-1]
      #     ...
      #   ]
      # --------------- chunk -----------------
      #   [
      #     [w12, w13, w14 ... w1n]
      #     [w22, w23, w24 ... w2n]
      #     ...
      #   ]
      # ]
      # after chunk, we have two chunks with the shape of (batch_size, max_seq_len - 1)
      indices = tags_pairs.permute(2, 0, 1).chunk(2)
     
      # (batch_size,)
      trans_score = self.transitionMatrix[indices].squeeze(0).sum(dim=-1)

      # STRAT => 0
      start_score = self.start_transitions[tags[:, 0]]
      # n-1 => END 
      stop_score = self.stop_transitions[tags[:, -1]]

      return feat_score + start_score + trans_score + stop_score

  
  def _partition_function(self, feats):
      """
       feats: 
        - (batch_size, max_seq_len, num_of_tag)
      """
      _, max_seq_len, num_tags = feats.shape

      # (batch_size, num_of_tag)
      a = feats[:, 0] + self.start_transitions.unsqueeze(0) 
    
      # [1, num_of_tag, num_of_tag] from -> to
      transitions = self.transitionMatrix.unsqueeze(0) 
      
      #   previous                      current word
      # (expand along x_axis)      (expand along y_axis) 
      # [ x_prev_1              [ x_cur_1, x_cur_2, x_cur_3
      #   x_prev_2 ...             ...
      #   x_prev_3 ]            ]

      # transition + previous + current_observation
      #   O B I
      # O          a0     O   B   I
      # B       +  a1  + 
      # I          a2    

      for i in range(1, max_seq_len):
        # (batch_size, num_of_tag)
        a = self._log_sum_exp(transitions + a.unsqueeze(-1) + feats[:, i].unsqueeze(1), 1)

      # (batch_size, num_of_tag) + (1, num_of_tag)
      return self._log_sum_exp(a + self.stop_transitions.unsqueeze(0), 1)

  
  def forward(self, feats):
      """
      feats: 
        - (batch_size, max_seq_len, num_of_tag)
      """
      _, max_seq_len, num_tags = feats.shape

      '''
      the first word of all sentencess
      '''
      # (batch_size, num_of_tag)
      pre_score = feats[:, 0] + self.start_transitions.unsqueeze(0)
     
      # ([1, num_of_tag, num_of_tag])
      transitions = self.transitionMatrix.unsqueeze(0)
     
      backpointers = []
      for i in range(1, max_seq_len): # loop each word
          # broadcast 
          # (batch_size, num_of_tag, 1)   (1, num_of_tag, num_of_tag)             
          # pre_score_0                     transitionMatrix

          # pre_score_1              + 
          
          # pre_score_2
          #   ...
          # pre_score_|num_of_tag|
          
          # (batch_size, num_of_tag)
          pre_score, pre_score_max_idx = (pre_score.unsqueeze(-1) + transitions).max(1)

          # [ (batch_size, num_of_tag), (batch_size, num_of_tag), ..., (batch_size, num_of_tag)]
          backpointers.append(pre_score_max_idx)

          pre_score = (pre_score + feats[:, i]) 

      # (batch_size, num_of_tag), (batch_size, 1)
      pre_score, last_best_id = (pre_score + self.stop_transitions.unsqueeze(0)).max(dim=1, keepdim=True)


      # [ (batch_size, 1) ]
      best_path = [last_best_id]
      for path_pointer in reversed(backpointers): # from the last word
          # [                                        [
          #   [1, 2, ... |num_of_tag| ]                [2]     
          #   [1, 2, ... |num_of_tag| ]   gather       [0]
          #   [1, 2, ... |num_of_tag| ]                [..]
          # ]                                        ]
          # the value in path_pointer indicates the best previous tag_id
          last_best_id = path_pointer.gather(1, last_best_id)
          best_path.append(last_best_id)

      best_path.reverse()

      # (batch_size, num_of_tag)
      return torch.cat(best_path, 1)



class BiLSTM_CRF_Batch(nn.Module):
  def __init__(
    self, 
    vocab_size, 
    tag2id,
    embedding_dim, 
    hidden_dim=256, 
    dropout=.5, 
    use_char_embed=False, 
    char_embedding_dim=25, 
    char_hidden_dim=64, 
    char2id=None, 
    pre_word_embedding=None,
    device=None,
  ):
    super(BiLSTM_CRF_Batch, self).__init__()

    self.device = device

    self.vocab_size = vocab_size
    self.embedding_dim = embedding_dim
    self.hidden_dim = hidden_dim
   
    self.tag2id = tag2id
    self.num_of_tag = len(tag2id)

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

    # context layer
    self.fc = nn.Linear(hidden_dim, self.num_of_tag)
    self.init_linear()

    self.crf = CRF(self.num_of_tag)


  def init_linear(self):
    init.xavier_normal_(self.fc.weight.data)
    init.normal_(self.fc.bias.data)


  def init_lstm(self, lstm):
    for param in lstm.parameters():
      if len(param.shape) >= 2:
          init.orthogonal_(param.data)
      else:
          init.normal_(param.data)


  def forward(self, inputs, input_chars=None, word_len_per_sent=None, perm_idx=None, seq_len_in_batch=None):
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
      max_seq_len = embed_words.shape[1]
      concat_char_embed_expand = torch.zeros_like(embed_words) # for padding word, use zero vector
      prev_seq_len = 0
      for i, seq_len in enumerate(seq_len_in_batch): # each sentence
        flat_char_embedding = torch.zeros(( max_seq_len, concat_char_embed.shape[1] ), dtype=torch.float)

        flat_char_embedding[:seq_len, :] = concat_char_embed[prev_seq_len:prev_seq_len+seq_len]

        concat_char_embed_expand[i] = flat_char_embedding

        prev_seq_len += seq_len

      # concat_char_embed_expand = concat_char_embed.reshape(embed_words.shape[0], -1, concat_char_embed.shape[1])

      total_embeds = torch.cat((embed_words, concat_char_embed_expand), -1)

      total_embeds = self.dropout(total_embeds)

      # (batch_size, max_seq_len, hidden_dim)
      lstm_out, _ = self.lstm(total_embeds)
    else:
      # (batch_size, max_seq_len, embed_dim)
      embed_words = self.word_embed(inputs)
    
      # (batch_size, max_seq_len, hidden_dim)
      lstm_out, _ = self.lstm(embed_words)


    lstm_out = self.dropout(lstm_out)

    # (batch_size, max_seq_len, num_of_tag)
    lstm_feats = self.fc(lstm_out)

    return lstm_feats


  def crf_decode(self, inputs, seq_len_in_batch=None):
    '''
    inputs: (batch_size, max_seq_len, num_of_tag)
    lengths: (batch_size, )
    '''
    prediction = self.crf(inputs)
    
    if seq_len_in_batch is not None:
      prediction = [ prediction[i, :length].data.cpu().numpy() 
        for i, length in enumerate(seq_len_in_batch) ]

    return prediction

  
  
  def crf_loss(self, inputs, true_pad_labels):
    """ 
      inputs: 
        - (batch_size, max_seq_len, num_of_tag)
      y: 
        - (batch_size, max_seq_len)
    """
    return self.crf.loss(inputs, true_pad_labels)



def main():
  batch_X = [
    "the wall street journal reported today that apple corporation made money".split(),
    "georgia tech is a university in georgia".split(),
  ]
  batch_y = [
    "B I I I O O O B I O O".split(),
    "B I O O O O B".split()
  ]

  tag_to_ix = {"O": 0, "B": 1, "I": 2,}

  word_to_ix = {
    '_PAD_': 0,
  }
  for sentence in batch_X:
      for word in sentence:
          if word not in word_to_ix:
              word_to_ix[word] = len(word_to_ix)

  
  # step 2
  def prepare_sequence(seq_batch, word_id, pad_id):
    seq_len_in_batch = [len(seq) for seq in seq_batch]
    max_seq_len = max(seq_len_in_batch)

    padded_seq_batch = torch.full( (len(seq_batch), max_seq_len), pad_id )    
    for i in range(len(seq_batch)):
      padded_seq_batch[i, :seq_len_in_batch[i]] = torch.LongTensor([word_id[w] for w in seq_batch[i]])

    return padded_seq_batch

  pad_batch_X = prepare_sequence(batch_X, word_to_ix, word_to_ix['_PAD_'])
  pad_batch_y = prepare_sequence(batch_y, tag_to_ix, 0)
  seq_len_in_batch = [len(seq) for seq in batch_X]


  print(pad_batch_X)
  print(pad_batch_y)



  # step 3
  model = BiLSTM_CRF_Batch(len(word_to_ix), tag_to_ix, 5, 4)
  optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)
  print(model)


  # step 4
  for epoch in range(1000):
    preds = model(pad_batch_X)
    loss = model.crf_loss(preds, pad_batch_y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


  with torch.no_grad():
    print('target ', pad_batch_y)
    preds = model(pad_batch_X)
    preds = model.crf_decode(preds, seq_len_in_batch)
    print(preds)



if __name__ == '__main__':
  main()
