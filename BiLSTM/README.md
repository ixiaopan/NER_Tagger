
# The Baseline BiLSTM + CRF Model

## Data

```bash
~ cd BiLSTM

# split data into train, valid, and test dataset
~ sh preprocess.sh

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

