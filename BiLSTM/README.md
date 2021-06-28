
# The Baseline BiLSTM + CRF Model

## Data

```bash
~ cd BiLSTM

# show help
# python build_dataset/build_word_dict.py -h

# split data into train, valid, and test dataset
~ python build_dataset.py --data_dir data/kaggle

# build vocabulary and word-id mapping
~ python build_word_dict.py --data_dir data/kaggle
```

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

