
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

# which is equivalent to the following two lines
~ sh baseline.sh toy

~ python train_baseline.py --train_data_dir='./data/toy'

~ python eval_baseline.py --data_dir='./data/toy'
```

### Pool
```bash

~ python train_baseline.py --train_data_dir='./data/pool' --model_param_dir='./experiments/pool'

~ python eval_baseline.py --data_dir='./data/mz' --model_param_dir='./experiments/pool'
```


### Pool-Init



## References

- [Multi-Domain Named Entity Recognition with Genre-Aware and Agnostic Inference](https://www.aclweb.org/anthology/2020.acl-main.750.pdf)


- [CS230 Tutorial](https://cs230.stanford.edu/blog/namedentity/)


- [Advanced Tutorial - PyTorch](https://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html)


- [ZubinGou/NER-BiLSTM-CRF-PyTorch](https://github.com/ZubinGou/NER-BiLSTM-CRF-PyTorch/tree/0146defefcc088b045016bafe5ea326fc52c7027)

