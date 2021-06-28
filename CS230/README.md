
# Re-implement CS230 NER

## Data

```bash
~ cd CS230

# https://www.kaggle.com/abhinavwalia95/entity-annotated-corpus/data
~ sh download.sh

# split data into train, valid, and test dataset
~ python build_dataset.py

# build vocabulary and word-id mapping
~ python build_word_dict.py
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
~ python train.py --data_dir 'data/toy' --model_dir 'exper/'

~ python evaluate.py --data_dir 'data/toy' --model_dir 'exper/'
```



## References

- [CS230 Tutorial](https://cs230.stanford.edu/blog/namedentity/)
