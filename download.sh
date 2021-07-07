# extract ontonotes
unzip data/SafeSend-u6etyJgw6CwkFGXt.zip -d data
unzip data/SafeSend-u6etyJgw6CwkFGXt/ontonotes_parsed.zip -d data/SafeSend-u6etyJgw6CwkFGXt

# download pre-trained GloVe
wget  http://nlp.stanford.edu/data/glove.6B.zip -P 'data/'
unzip data/glove.6B.zip -d data/glove.6B

# toy test data
wget 'https://raw.githubusercontent.com/cs230-stanford/cs230-code-examples/master/pytorch/nlp/data/small/test/labels.txt' -P 'data/toy/test'
wget 'https://raw.githubusercontent.com/cs230-stanford/cs230-code-examples/master/pytorch/nlp/data/small/test/sentences.txt' -P 'data/toy/test'

# toy train data
wget 'https://raw.githubusercontent.com/cs230-stanford/cs230-code-examples/master/pytorch/nlp/data/small/train/labels.txt' -P 'data/toy/train'
wget 'https://raw.githubusercontent.com/cs230-stanford/cs230-code-examples/master/pytorch/nlp/data/small/train/sentences.txt' -P 'data/toy/train'

# toy valid data
wget 'https://raw.githubusercontent.com/cs230-stanford/cs230-code-examples/master/pytorch/nlp/data/small/val/labels.txt' -P 'data/toy/valid'
wget 'https://raw.githubusercontent.com/cs230-stanford/cs230-code-examples/master/pytorch/nlp/data/small/val/sentences.txt' -P 'data/toy/valid'
