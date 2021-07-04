
# The Baseline BiLSTM + CRF Model

## Data Preprocessing

```bash
~ cd BiLSTM

# download toy dataset 
~ sh download.sh

# clean ontonotes dataset & extract each domain
~ sh clean.sh

# split each domain dataset into train, valid, and test
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
~ python build_onto_profile.py --data_dir='./data/simple' --use_pre_trained=false --augment_vocab_from_glove=false

~ python train_baseline.py --data_dir='./data/simple'

~ python eval_baseline.py --data_dir='./data/simple'
```


## References

- [Neural Architectures for Named Entity Recognition - Github](https://github.com/glample/tagger)

- [CS230 Tutorial](https://cs230.stanford.edu/blog/namedentity/)

- [Advanced Tutorial - PyTorch](https://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html)

- [ZubinGou/NER-BiLSTM-CRF-PyTorch](https://github.com/ZubinGou/NER-BiLSTM-CRF-PyTorch/tree/0146defefcc088b045016bafe5ea326fc52c7027)

