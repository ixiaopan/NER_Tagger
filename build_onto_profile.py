"""
Build vocabulary & tag_id & char_id
"""

import argparse
from utils import utils


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='./data/toy', help="Directory containing the dataset")
parser.add_argument('--min_word_freq', default=1, help="The minimum frequency of a word")
parser.add_argument('--use_char_embed', default=True, action='store_true', help="Character-level embedding")
parser.add_argument('--use_pre_trained', default=True, action='store_true', help="Whether to use pre-trained word embedding")
parser.add_argument('--glove_word_dim', default=300, help="The dimension of GloVe word vector")


if __name__ == '__main__':
  args = parser.parse_args()
  
  utils.build_ner_profile(
    args.data_dir, 
    min_word_freq=args.min_word_freq, 
    use_char_embed=args.use_char_embed,
    use_pre_trained=args.use_pre_trained,
    glove_word_dim=int(args.glove_word_dim),
  )
