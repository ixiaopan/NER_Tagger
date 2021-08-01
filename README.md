
# Multi-Domain Named Entity Recognition


## Virtual Environment
```bash
# create workspace
~ python3 -m venv venv

# activate workspace
~ source venv/bin/activate

~ export PYTHONPATH="${PYTHONPATH}:/Users/wuxiaopan/work/NER_Tagger"

# ~ sys.path # check it out

# leave the virtual env
~ deactivate

# export dependencies
~ pipreqs server

# install dependencies
~ pip install -r requirements.txt
```


## Data Preprocessing

```bash

# extract ontonotest & download GloVe
~ sh download.sh

# clean ontonotes dataset & extract each domain
~ sh clean.sh
```

- bc
  - train
    - sentences.txt
    - labels.txt
  - valid
    - sentences.txt
    - labels.txt
  - test
    - sentences.txt
    - labels.txt



## Model Training and Evaluation

### Baseline

```bash

~ sh baseline.sh toy

# or step train_eval
~ python batch_train_baseline.py --train_data_dir='./data/toy'

~ python batch_eval_baseline.py --data_dir='./data/toy'
```


### Pool
```bash
# data pooling aggregated data from all domains
~ python batch_train_baseline.py --train_data_dir='./data/pool' --model_param_dir='./experiments/pool'

~ python batch_eval_baseline.py --data_dir='./data/tc' --model_param_dir='./experiments/pool'

# data pooling aggregated data from leave-one-domain-out domains
~ python batch_train_baseline.py --train_data_dir='./data/pool_bc' --model_param_dir='./experiments/pool_bc'

~ python batch_eval_baseline.py --data_dir='./data/bc' --model_param_dir='./experiments/pool_bc'
```

### Feature Type

```bash
# sent length [5, 10, 30, 60, max]
~ python split_by_sent_len.py --domain='bc'

~ python batch_eval_baseline.py --data_dir='./data/bc' --split_type='test_sent_5' --model_param_dir='./experiments/pool_bc'

# rare word [5, 25, 50, 75, 100]
~ python split_by_rare_word.py --domain='bc'

~ python batch_eval_baseline.py --data_dir='./data/bc' --split_type='test_rare_5' --model_param_dir='./experiments/pool_bc'
```


### Pool-Init

```bash
~ python batch_train_pool_init.py --model_param_dir='./experiments/pool_init'

~ python batch_eval_baseline.py --data_dir='./data/tc' --model_param_dir='./experiments/pool_init'
```


## References

- [Multi-Domain Named Entity Recognition with Genre-Aware and Agnostic Inference](https://www.aclweb.org/anthology/2020.acl-main.750.pdf)


- [CS230 Tutorial](https://cs230.stanford.edu/blog/namedentity/)


- [Advanced Tutorial - PyTorch](https://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html)


- [ZubinGou/NER-BiLSTM-CRF-PyTorch](https://github.com/ZubinGou/NER-BiLSTM-CRF-PyTorch/tree/0146defefcc088b045016bafe5ea326fc52c7027)

