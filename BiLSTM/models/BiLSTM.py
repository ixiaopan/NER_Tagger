'''
The baseline ner model
'''

import torch
import torch.nn as nn
from torch.nn import init
import numpy as np


torch.manual_seed(2)


def argmax(vec):
  _, idx = torch.max(vec, 1)
  return idx.item()


def prepare_sequence(seq, word_id):
  idxs = [word_id[w] for w in seq]
  return torch.tensor(idxs, dtype=torch.long)


def log_sum_exp(vec):
  max_score = vec[0, argmax(vec)]

  max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])

  return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


def init_lstm(input_lstm):
  """
  Initialize lstm
  """
  for param in input_lstm.parameters():
      if len(param.shape) >= 2:
          init.orthogonal_(param.data)
      else:
          init.normal_(param.data)


def init_linear(input_linear):
    """
    Initialize linear transformation
    """
    init.xavier_normal_(input_linear.weight.data)
    init.normal_(input_linear.bias.data)


START_TAG = '_START_'
STOP_TAG = '_STOP_'


class BiLSTM_CRF(nn.Module):
  def __init__(
    self, 
    vocab_size, tag2id, embedding_dim, hidden_dim=256, dropout=.5, 
    use_char_embed=False, char_embedding_dim=25, char_hidden_dim=64, char2id=None, 
    pre_word_embedding=None
  ):
    super(BiLSTM_CRF, self).__init__()
    
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
      init_lstm(self.char_lstm)

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
    init_lstm(self.lstm)

    # context layer
    self.fc = nn.Linear(hidden_dim, self.num_of_tag)
    init_linear(self.fc)

    # CRF layer transition score
    # T_{ij} transfer from i to j, (k+2, k+2)
    self.transitionMatrix = nn.Parameter(torch.randn(self.num_of_tag, self.num_of_tag))
    # never transfer to the start tag
    self.transitionMatrix.data[:, tag2id[START_TAG]] = -10000
    # never transfer from the stop tag
    self.transitionMatrix.data[tag2id[STOP_TAG], :] = -10000


  def _forward_alg(self, feats):
    init_alphas = torch.full((1, self.num_of_tag), -1000.)
    
    init_alphas[0][self.tag2id[START_TAG]] = 0

    previous_score = init_alphas

    for feat in feats: # each word
      all_possible_tag_score = []
      for next_tag in range(self.num_of_tag):
        emit_score = feat[next_tag].view(1, -1).expand(1, self.num_of_tag)
        
        trans_score = self.transitionMatrix[:, next_tag].view(1, -1)
        
        next_tag_score = previous_score + trans_score + emit_score
        
        all_possible_tag_score.append(log_sum_exp(next_tag_score).view(1))
    
      previous_score = torch.cat(all_possible_tag_score).view(1, -1)
    
    total_path_score = previous_score + self.transitionMatrix[:, self.tag2id[STOP_TAG]]
    
    return log_sum_exp(total_path_score)


  def _get_lstm_features(
    self, 
    inputs, input_chars=None, 
    word_len_in_batch=None, perm_idx=None, 
    device=None
  ):
    '''
    inputs: 
      (batch_size, max_seq_len)
    input_chars: 
      (batch_size*max_seq_len, max_word_len)
    word_len_in_batch: 
      (batch_size*max_seq_len,)
    perm_idx: 
      (batch_size*max_seq_len,)
    '''

    if self.use_char_embed:
      # (batch_size*max_seq_len, max_word_len, char_embedding_dim)
      embed_chars = self.char_embed(input_chars)
      # print('embed_chars shape ', embed_chars.shape, )

      # (sum(word_len_in_batch), char_embedding_dim)
      packed_char_inputs = torch.nn.utils.rnn.pack_padded_sequence(embed_chars, word_len_in_batch, batch_first=True)
      # print('packed_char_inputs shape ', packed_char_inputs.data.shape)

      # (sum(word_len_in_batch), char_hidden_dim)
      lstm_char_out, _ = self.char_lstm(packed_char_inputs)
      # print('lstm_char_out shape ', lstm_char_out.data.shape)

      # char_outpus, (batch_size*max_seq_len, max_word_len, char_hidden_dim)
      # char_input_sizes: equal to word_len_in_batch
      char_outputs, char_input_sizes = torch.nn.utils.rnn.pad_packed_sequence(lstm_char_out, batch_first=True)
      # print('char_outputs shape ', char_outputs.shape)
      
      # concatenate the last output of BiLSTM
      # (batch_size*max_seq_len, char_hidden_dim)
      concat_char_embed_sorted = torch.zeros(
        (char_outputs.size(0), char_outputs.size(2)), 
        dtype=torch.float
      )
      
      for i, index in enumerate(char_input_sizes):
        concat_char_embed_sorted[i] = torch.cat((
            char_outputs[ i, index - 1, :(self.char_hidden_dim // 2) ],
            char_outputs[ i, 0, (self.char_hidden_dim // 2): ],
        ))

      # print('concat_char_outputs shape ', concat_char_embed_sorted.shape)

      # the obtained concat_char_embedding is sorted in descending order by seq_len
      # however, the inputs is the original inputs, so we need map them correctly so as to concate them later
      concat_char_embed = torch.zeros_like(concat_char_embed_sorted)
      for i in range(concat_char_embed.size(0)):
        concat_char_embed[perm_idx[i]] = concat_char_embed_sorted[i]
      concat_char_embed.to(device)

    
    if self.use_char_embed:
      # (batch_size, seq_len, embed_dim)
      embed_words = self.word_embed(inputs)
      # print('word embedding shape ', embed_words.shape)

      total_embeds = torch.cat((
        embed_words, 
        concat_char_embed.reshape(embed_words.shape[0], -1, concat_char_embed.shape[1])
      ), 2)
      # print('total_embeds shape ', embed_words.shape)
      
      total_embeds = self.dropout(total_embeds)
      
      # lstm_out: (batch_size, seq_len, hidden_dim)
      lstm_out, _ = self.lstm(total_embeds)

      # print('lstm_out shape ', lstm_out.shape)
    else:
      # (batch_size, seq_n, embed_dim)
      embed_words = self.word_embed(inputs)
      # embed_words = embed_words.view(embed_words.shape[0], -1, self.embedding_dim)
      lstm_out, _ = self.lstm(embed_words)
    

    # (batch_size*seq_len, hidden_dim)
    lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)   

    lstm_out = self.dropout(lstm_out)
   
    # emission score
    # B-Per, I-Per, B-LOC, I-LOC, O
    # 0.3,    0.2,   0.2    0.2   0.1
    # (batch_size*seq_len, num_of_tag)
    lstm_feats = self.fc(lstm_out)
    # print('lstm_feats shape ', lstm_feats.shape)

    return lstm_feats
  
  
  def _score_sentence(self, feats, tags):
    '''
    predict sequence score based on curren predicted feats
    feats(batch_size*seq_len, num_of_tag): predicted label
    tags(batch_size, seq_len): ground truth

    x_{i, label}: the score when the ith word is labelled by 'label'
    for x_{i, start}, x_{i, stop}, the score is 0

    L(y|x) = \frac{P_{realpath}}{\sum_i^N P_i} = \frac{Score(x, y)}{\sum_y' Score(x, y')}
    -log L(y|x) = log P_total - log P_realpath
    score(x, y) = \sum_i log trans(y_{i-1}, y_i) + log emit(x_i|y_i)
    '''

    # step 1
    batch_size = tags.shape[0]
   
    batch_feats = feats.reshape(batch_size, -1, feats.shape[1]) # (batch_size, seq_len, num_of_tag)
    # print('batch_feats shape', batch_feats.shape)

    batch_mask = tags >= 0
    batch_tags = [ tags[j][ tags[j] >= 0 ] for j in range(batch_size) ] 

    # step 2
    batch_score = torch.zeros((batch_size, 1))
    
    for j in range(batch_size):
      tags_per_sent = batch_tags[j]

      # add 'start'
      tags_per_sent = torch.cat([torch.tensor([self.tag2id[START_TAG]], dtype=torch.long), tags_per_sent])
      
      # real seq_len
      feat_per_sent = batch_feats[j][batch_mask[j]]

      # print('per sent in batch', tags_per_sent.shape, feat_per_sent.shape)

      for i, feat in enumerate(feat_per_sent): # loop each word
        batch_score[j] = batch_score[j] + \
          self.transitionMatrix[tags_per_sent[i], tags_per_sent[i+1]] + feat[tags_per_sent[i+1]]

      # add 'stop'
      batch_score[j] = batch_score[j] + \
        self.transitionMatrix[tags_per_sent[-1], self.tag2id[STOP_TAG]]

    return batch_score

  def _batch_forward_alg(self, batch_feats, batch_tags):
    batch_size = batch_tags.shape[0]
    
    # (batch_size, seq_len, num_of_tag)
    batch_feats = batch_feats.reshape(batch_size, -1, batch_feats.shape[1]) 
    
    batch_mask = batch_tags >= 0
    
    batch_score = torch.zeros((batch_size, 1))

    for j in range(batch_size):
      feats_per_sent = batch_feats[j][batch_mask[j]]

      batch_score[j] = self._forward_alg(feats_per_sent)

    return batch_score


  def neg_log_likelihood(self, inputs, batch_tags, input_chars=None, word_len_in_batch=None, perm_idx=None, device=None):
    '''
    inputs: (batch_size, max_seq_len)
    batch_tags: (batch_size, max_seq_len)
    input_chars: (batch_size*max_seq_len, max_word_len)
    word_len_in_batch: word_len in batch
    '''
    batch_feats = self._get_lstm_features(inputs, input_chars, word_len_in_batch, perm_idx, device)

    forward_score = self._batch_forward_alg(batch_feats, batch_tags)

    gold_score = self._score_sentence(batch_feats, batch_tags)
    
    return (forward_score - gold_score).sum() / batch_tags.shape[0] 
  
  
  # inference
  def _viterbi_decode(self, feats):
    backpointers = []

    init_score = torch.full((1, self.num_of_tag), -10000)
    init_score[0][self.tag2id[START_TAG]] = 0

    previous_score = init_score

    for feat in feats:
      bptrs_t = []
      viterbi_tag_score = [] # score when word x_i is labelled by next_tag

      for next_tag in range(self.num_of_tag):
        # emission score stay the same across all the tags
        next_tag_score = previous_score + self.transitionMatrix[:, next_tag].view(1, -1)
        
        # the best previous tag to the current 'next_tag'
        best_tag_id = argmax(next_tag_score) 
        bptrs_t.append(best_tag_id)

        # the score when labelled as 'next_tag', (1, 1)
        viterbi_tag_score.append(next_tag_score[0][best_tag_id].view(1))

      # (1, num_of_tag)
      previous_score = (torch.cat(viterbi_tag_score) + feat).view(1, -1)
      
      # ['start_tag', 'start_tag', 'start_tag', ..., tag_k]
      backpointers.append(bptrs_t) # (len(sentence), num_of_tag)

    terminal_score = previous_score + self.transitionMatrix[:, self.tag2id[STOP_TAG]]
    last_best_tag_id = argmax(terminal_score)
    path_score = terminal_score[0][last_best_tag_id]

    best_path = [last_best_tag_id]
    for path_pointer in reversed(backpointers):
      last_best_tag_id = path_pointer[last_best_tag_id]
      best_path.append(last_best_tag_id)

    start = best_path.pop()
    assert start == self.tag2id[START_TAG]

    best_path.reverse()
    return path_score, best_path



  def _batch_viterbi_decode(self, batch_feats, batch_size):

    batch_feats = batch_feats.reshape(batch_size, -1, batch_feats.shape[1]) 

    batch_score = []

    for j in range(batch_size):
      feats_per_sent = batch_feats[j]
      batch_score.append(self._viterbi_decode(feats_per_sent))

    return batch_score


  def forward(self, inputs, input_chars=None, word_len_in_batch=None, perm_idx=None, device=None):
    '''
    inputs: batch_size, max_seq_len
    input_char: (batch_size*max_seq_len, max_word_len)
    word_len_in_batch: word_len in batch
    '''
    batch_feats = self._get_lstm_features(inputs, input_chars, word_len_in_batch, perm_idx, device)

    batch_score = self._batch_viterbi_decode(batch_feats, inputs.shape[0])

    # (batch_size, (score, tag_seq))
    return batch_score
