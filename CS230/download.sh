rm -rf data/*

toy test data
wget 'https://raw.githubusercontent.com/cs230-stanford/cs230-code-examples/master/pytorch/nlp/data/small/test/labels.txt' -P 'data/toy/test'
wget 'https://raw.githubusercontent.com/cs230-stanford/cs230-code-examples/master/pytorch/nlp/data/small/test/sentences.txt' -P 'data/toy/test'

# toy train data
wget 'https://raw.githubusercontent.com/cs230-stanford/cs230-code-examples/master/pytorch/nlp/data/small/train/labels.txt' -P 'data/toy/train'
wget 'https://raw.githubusercontent.com/cs230-stanford/cs230-code-examples/master/pytorch/nlp/data/small/train/sentences.txt' -P 'data/toy/train'

# toy valid data
wget 'https://raw.githubusercontent.com/cs230-stanford/cs230-code-examples/master/pytorch/nlp/data/small/val/labels.txt' -P 'data/toy/valid'
wget 'https://github.com/cs230-stanford/cs230-code-examples/blob/master/pytorch/nlp/data/small/val/sentences.txt' -P 'data/toy/valid'

# download
kaggle datasets download -d abhinavwalia95/entity-annotated-corpus -p 'data/kaggle'
unzip './data/kaggle/entity-annotated-corpus.zip' -d ./data/kaggle
