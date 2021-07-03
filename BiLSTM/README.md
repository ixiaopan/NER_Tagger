
# The Baseline BiLSTM + CRF Model

## Data Preprocessing

```bash
~ cd BiLSTM

# clean ontonotes dataset & extract each domain
~ sh clean.sh

# split each domain dataset into train, valid, and test
~ python build_onto_dataset.py --domain bc

# build vocabulary, word_id, tag_id, pre_trained word embedding for each domain
~ python build_onto_profile.py --data_dir='./data/toy' --use_pre_trained=1
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
~ python train.py 
~ python evaluate.py
```


## References

- [Neural Architectures for Named Entity Recognition - Github](https://github.com/glample/tagger)

