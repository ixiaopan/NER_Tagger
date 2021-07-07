'''
The baseline ner model
@source: https://github.com/ZubinGou/NER-BiLSTM-CRF-PyTorch/blob/0146defefcc088b045016bafe5ea326fc52c7027/src/model.py
'''

import torch
import torch.nn as nn
from torch.nn import init
import numpy as np
from utils import utils

torch.manual_seed(2)



def argmax(vec):
  _, idx = torch.max(vec, 1)
  return idx.item()



def log_sum_exp(vec):
  max_score = vec.max(dim=0, keepdim=True).values
  return max_score + torch.log(torch.exp(vec - max_score).sum(dim=0, keepdim=True))

  # per epoch takes at least 10 minutes using GPU
  # max_score = vec[0, argmax(vec)]
  # max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
  # return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


def init_lstm(input_lstm):
  for param in input_lstm.parameters():
    if len(param.shape) >= 2:
        init.orthogonal_(param.data)
    else:
        init.normal_(param.data)


def init_linear(input_linear):
  init.xavier_normal_(input_linear.weight.data)
  init.normal_(input_linear.bias.data)


START_TAG = '_START_'
STOP_TAG = '_STOP_'


class BiLSTM_CRF(nn.Module):
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
    super(BiLSTM_CRF, self).__init__()
    
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
    init_alphas = torch.full((1, self.num_of_tag), -1000., device=self.device)
    init_alphas[0][self.tag2id[START_TAG]] = 0

    for feat in feats:
      init_alphas = log_sum_exp(init_alphas.T + feat.unsqueeze(0) + self.transitionMatrix)
    
    return log_sum_exp(init_alphas.T + self.transitionMatrix[:, self.tag2id[STOP_TAG]].view(-1, 1)).flatten()[0]


    # init_alphas = torch.full((1, self.num_of_tag), -1000., device=self.device)
    # init_alphas[0][self.tag2id[START_TAG]] = 0
    # previous_score = init_alphas

    # for feat in feats: # each word
    #   all_possible_tag_score = []
    #   '''
    #   [           [         [
    #     alpha[0]     1 2
    #     alpha[1]  +  1 2   +   transition[:,0], transition[:,1] 
    #     alpha[2]     1 2   
    #   '''
    #   for next_tag in range(self.num_of_tag):
    #     emit_score = feat[next_tag].view(1, -1).expand(1, self.num_of_tag)
    #     trans_score = self.transitionMatrix[:, next_tag].view(1, -1)
    #     next_tag_score = previous_score + trans_score + emit_score    

    #     all_possible_tag_score.append(log_sum_exp(next_tag_score).view(1))

    #   previous_score = torch.cat(all_possible_tag_score).view(1, -1)
    
    # total_path_score = previous_score + self.transitionMatrix[:, self.tag2id[STOP_TAG]]
    
    # return log_sum_exp(total_path_score)



  def _get_lstm_features(
    self, 
    inputs, 
    input_chars=None, 
    word_len_per_sent=None, 
    perm_idx=None
  ):
    '''
    inputs, (seq_n)
    
    input_chars, (seq_n, max_seq_len)
    # [
    #   [ 6  9  8  4  1 11 12 10 ] # 8
    #   [ 7  3  2  5 13  7  0  0 ] # 6
    #   [ 12  5  8 14  0  0  0  0 ] # 4
    # ]
    '''

    if self.use_char_embed:
      # (seq_n, max_seq_len, char_embedding_dim)
      embed_chars = self.char_embed(input_chars)

      # (sum_of_seq_len, char_embedding) => (8+6+4, char_embedding_dim)
      packed_char_inputs = torch.nn.utils.rnn.pack_padded_sequence(embed_chars, word_len_per_sent, batch_first=True)

      # (sum_of_seq_len, char_hidden_dim) => (8+6+4, char_hidden_dim)
      lstm_char_out, _ = self.char_lstm(packed_char_inputs)

      # char_outpus, (seq_n, max_seq_len, char_hidden_dim)
      # char_input_sizes: equal to word_len_per_sent
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

  
    if self.use_char_embed:
      # (seq_n, embed_dim)
      embed_words = self.word_embed(inputs)
      
      total_embeds = torch.cat((embed_words, concat_char_embed), 1)
      # (batch_size=1, seq_n, word_embedding) add batch_size=1 at the 0th-dimension
      total_embeds = total_embeds.unsqueeze(0)
      
      total_embeds = self.dropout(total_embeds)
      # lstm_out: (batch_size=1, seq_n, hidden_dim)
      
      lstm_out, _ = self.lstm(total_embeds)
    else:
      # (batch_size, seq_n, embed_dim)
      embed_words = self.word_embed(inputs).view(1, -1, self.embedding_dim)
      
      lstm_out, _ = self.lstm(embed_words)
    

    # (batch_size*seq_n, hidden_dim)
    lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
    
    lstm_out = self.dropout(lstm_out)
   
    # emission score
    # B-Per, I-Per, B-LOC, I-LOC, O
    # 0.3,    0.2,   0.2    0.2   0.1
    # (batch_size*seq_n, num_of_tag)
    lstm_feats = self.fc(lstm_out)

    return lstm_feats
  
  
  def _score_sentence(self, feats, tags):
    '''
    predict sequence score based on curren predicted feats

    @params
      - feats(batch_size*seq_n, num_of_tag): predicted label
      - tags(batch_size, seq_n): ground truth
    @return
      - (1, )

    x_{i, label}: the score when the ith word is labelled by 'label'
    for x_{i, start}, x_{i, stop}, the score is 0

    L(y|x) = \frac{P_{realpath}}{\sum_i^N P_i} = \frac{Score(x, y)}{\sum_y' Score(x, y')}
    -log L(y|x) = log P_total - log P_realpath
    score(x, y) = \sum_i log trans(y_{i-1}, y_i) + log emit(x_i|y_i)
    '''

    # matrix-form
    # add 'start'
    pad_start_tags = torch.cat([ torch.tensor([self.tag2id[START_TAG]], dtype=torch.long).to(self.device), tags ])
    # add 'end'
    pad_stop_tags = torch.cat([ tags, torch.tensor([self.tag2id[START_TAG]], dtype=torch.long).to(self.device) ])
    
    return torch.sum(self.transitionMatrix[pad_start_tags, pad_stop_tags]) + torch.sum(feats[range(len(tags)), tags])


    # score = torch.zeros(1).to(self.device)
    # # add 'start'
    # tags = torch.cat([torch.tensor([self.tag2id[START_TAG]], dtype=torch.long), tags])
    # for i, feat in enumerate(feats): # loop each word
    #   score = score + self.transitionMatrix[tags[i], tags[i+1]] + feat[tags[i+1]]
    # # add 'stop'
    # score = score + self.transitionMatrix[tags[-1], self.tag2id[STOP_TAG]]
    # return score



  def neg_log_likelihood(self, inputs, tags, input_chars=None, word_len_per_sent=None, perm_idx=None):
    feats = self._get_lstm_features(inputs, input_chars, word_len_per_sent, perm_idx)

    forward_score = self._forward_alg(feats)

    gold_score = self._score_sentence(feats, tags)

    return forward_score - gold_score
  
  

  # inference
  def _viterbi_decode(self, feats):
    backpointers = []

    init_score = torch.full((1, self.num_of_tag), -10000, device=self.device)
    init_score[0][self.tag2id[START_TAG]] = 0

    for feat in feats:
      next_tag_score = init_score.T + feat.unsqueeze(0) + self.transitionMatrix
      
      # (len(sentence), num_of_tag)
      backpointers.append(torch.argmax(next_tag_score, dim=0))

      init_score = log_sum_exp(next_tag_score)

    terminal_score = init_score.T + self.transitionMatrix[:, [ self.tag2id[STOP_TAG]]]
    last_best_tag_id = torch.argmax(terminal_score.view(-1)).item()
    best_path = [last_best_tag_id]

    for path_pointer in reversed(backpointers):
      last_best_tag_id = path_pointer[last_best_tag_id].item()
      best_path.append(last_best_tag_id)

    start = best_path.pop()
    assert start == self.tag2id[START_TAG]

    best_path.reverse()
    return log_sum_exp(terminal_score), best_path


    # backpointers = []
    # init_score = torch.full((1, self.num_of_tag), -10000, device=self.device)
    # init_score[0][self.tag2id[START_TAG]] = 0
    # previous_score = init_score

    # for feat in feats:
    #   bptrs_t = []
    #   viterbi_tag_score = [] # score when word x_i is labelled by next_tag

    #   for next_tag in range(self.num_of_tag):
    #     # emission score stay the same across all the tags
    #     next_tag_score = previous_score + self.transitionMatrix[:, next_tag].view(1, -1)
        
    #     # the best previous tag to the current 'next_tag'
    #     best_tag_id = argmax(next_tag_score) 
    #     bptrs_t.append(best_tag_id)

    #     # the score when labelled as 'next_tag', (1, 1)
    #     viterbi_tag_score.append(next_tag_score[0][best_tag_id].view(1))

    #   # (1, num_of_tag)
    #   previous_score = (torch.cat(viterbi_tag_score) + feat).view(1, -1)
      
    #   # ['start_tag', 'start_tag', 'start_tag', ..., tag_k]
    #   backpointers.append(bptrs_t) # (len(sentence), num_of_tag)

    # terminal_score = previous_score + self.transitionMatrix[:, self.tag2id[STOP_TAG]]
    # last_best_tag_id = argmax(terminal_score)
    # path_score = terminal_score[0][last_best_tag_id]

    # best_path = [last_best_tag_id]
    # for path_pointer in reversed(backpointers):
    #   last_best_tag_id = path_pointer[last_best_tag_id]
    #   best_path.append(last_best_tag_id)


    # start = best_path.pop()
    # assert start == self.tag2id[START_TAG]

    # best_path.reverse()
    # return path_score, best_path


  def forward(self, inputs, input_chars=None, word_len_per_sent=None, perm_idx=None):
    feats = self._get_lstm_features(inputs, input_chars, word_len_per_sent, perm_idx)

    score, tag_seq = self._viterbi_decode(feats)

    return score, tag_seq
