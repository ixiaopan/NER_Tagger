
# Re-implement CS230 NER


## Virtual Environment
```bash
# create workspace
~ python3 -m venv venv

# activate workspace
~ source venv/bin/activate

# leave the virtual env
~ deactivate

# export dependencies
~ pipreqs server

# install dependencies
~ pip install -r requirements.txt
```

## Data

```bash
~ cd CS230

# https://www.kaggle.com/abhinavwalia95/entity-annotated-corpus/data
~ sh download.sh

~ python build_dataset.py
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

## Model Traning and Evaluation

```bash

~ python train.py --data_dir 'data/toy' --model_dir 'exper/'

~ python evaluate.py --data_dir 'data/toy' --model_dir 'exper/'
```

## References

- [CS230 Tutorial](https://cs230.stanford.edu/blog/namedentity/)
