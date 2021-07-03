"""
Build vocabulary & label and summary the statistics
"""

import argparse
from utils import utils


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='./data/toy', help="Directory containing the dataset")
parser.add_argument('--min_word_freq', default=2, help="The minimum frequency of a word")
parser.add_argument('--use_pre_trained', default=0, help="Whether to use pre-trained word embedding")
parser.add_argument('--glove_word_dim', default=50, help="The dimension of GloVe word vector")
parser.add_argument('--augment_vocab_from_glove', default=False, help="Augment vocabulary from glove or test dataset")

if __name__ == '__main__':
  args = parser.parse_args()
  
  utils.build_ner_profile(
    args.data_dir, 
    min_word_freq=args.min_word_freq, 
    use_pre_trained=args.use_pre_trained,
    glove_word_dim=args.glove_word_dim,
    augment_vocab_from_glove=args.augment_vocab_from_glove
  )
