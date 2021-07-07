
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

# split each domain dataset into train, valid, and test
# if domain is not specified, it will loop all domain
~ python build_onto_dataset.py --domain='bc'
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

```bash
# build vocabulary, word_id, tag_id, pre_trained word embedding for each domain
~ python build_onto_profile.py --data_dir='./data/toy' --use_pre_trained

~ python train_baseline.py --data_dir='./data/toy'

~ python eval_baseline.py --data_dir='./data/toy'
```




## References

- [Multi-Domain Named Entity Recognition with Genre-Aware and Agnostic Inference](https://www.aclweb.org/anthology/2020.acl-main.750.pdf)


- [CS230 Tutorial](https://cs230.stanford.edu/blog/namedentity/)


- [Advanced Tutorial - PyTorch](https://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html)


- [ZubinGou/NER-BiLSTM-CRF-PyTorch](https://github.com/ZubinGou/NER-BiLSTM-CRF-PyTorch/tree/0146defefcc088b045016bafe5ea326fc52c7027)

