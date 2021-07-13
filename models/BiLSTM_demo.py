"""
@reference: 
  - https://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html
"""

import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(1)

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


START_TAG = '_START_'
STOP_TAG = '_STOP_'
embedding_dim = 5
hidden_dim = 4


class BiLSTM_CRF(nn.Module):
  def __init__(self, vocab_size, tag2id, embedding_dim, hidden_dim=256, dropout=.5):
    super(BiLSTM_CRF, self).__init__()

    self.vocab_size = vocab_size
    self.embedding_dim = embedding_dim
    self.hidden_dim = hidden_dim
   
    self.tag2id = tag2id
    self.num_of_tag = len(tag2id)
   
    self.word_embed = nn.Embedding(vocab_size, embedding_dim)
    self.dropout = nn.Dropout(dropout)
    self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers = 1, bidirectional=True, batch_first=True)
    self.fc = nn.Linear(hidden_dim, self.num_of_tag)

    # crf layer
    # T_{ij} transfer from i to j, (k+2, k+2)
    self.transitionMatrix = nn.Parameter(torch.randn(self.num_of_tag, self.num_of_tag))
    # never transfer to the start tag, transfer from the stop tag,
    self.transitionMatrix.data[:, tag2id[START_TAG]] = -10000
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


  def _get_lstm_features(self, inputs):
    '''
    inputs(batch_size, seq_n): batch_n sentences
    '''

    # (batch_size, seq_n, embed_dim)
    embed_words = self.word_embed(inputs).view(1, -1, self.embedding_dim)

    # lstm_out: (batch_size, seq_n, hidden_dim)
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
    feats(batch_size*seq_n, num_of_tag): predicted label
    tags(batch_size, seq_n): ground truth

    x_{i, label}: the score when the ith word is labelled by 'label'
    for x_{i, start}, x_{i, stop}, the score is 0

    L(y|x) = \frac{P_{realpath}}{\sum_i^N P_i} = \frac{Score(x, y)}{\sum_y' Score(x, y')}
    -log L(y|x) = log P_total - log P_realpath
    score(x, y) = \sum_i log trans(y_{i-1}, y_i) + log emit(x_i|y_i)
    '''

    score = torch.zeros(1)

    # add 'start'
    tags = torch.cat([torch.tensor([self.tag2id[START_TAG]], dtype=torch.long), tags])

    for i, feat in enumerate(feats): # loop each word
      score = score + self.transitionMatrix[tags[i], tags[i+1]] + feat[tags[i+1]]

    # add 'stop'
    score = score + self.transitionMatrix[tags[-1], self.tag2id[STOP_TAG]]

    return score


  def neg_log_likelihood(self, inputs, tags):
    feats = self._get_lstm_features(inputs)

    forward_score = self._forward_alg(feats)

    gold_score = self._score_sentence(feats, tags)

    return forward_score - gold_score
  
  
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


  def forward(self, inputs):
    feats = self._get_lstm_features(inputs)

    score, tag_seq = self._viterbi_decode(feats)
    
    return score, tag_seq



training_data = [
  (
    "the wall street journal reported today that apple corporation made money".split(),
    "B I I I O O O B I O O".split()
  ), 
  (
    "georgia tech is a university in georgia".split(),
    "B I O O O O B".split()
  )
]

word_to_ix = {}
for sentence, tags in training_data:
    for word in sentence:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)

tag_to_ix = {"B": 0, "I": 1, "O": 2, START_TAG: 3, STOP_TAG: 4}

model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, embedding_dim, hidden_dim)
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

print(model)

with torch.no_grad():
  precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
  precheck_tags = torch.tensor([tag_to_ix[t] for t in training_data[0][1]], dtype=torch.long)
  print(model(precheck_sent))


for epoch in range(200):
  for sentence, tags in training_data:
    model.zero_grad()

    sent_id = prepare_sequence(sentence, word_to_ix)
    targets = torch.tensor([ tag_to_ix[t] for t in tags ], dtype=torch.long)
    loss = model.neg_log_likelihood(sent_id, targets)
    
    loss.backward()
    optimizer.step()


with torch.no_grad():
  precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
  
  precheck_tags = [tag_to_ix[t] for t in training_data[0][1]]
  
  print('target ', precheck_tags)
  print(model(precheck_sent))
