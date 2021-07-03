"""
Build vocabulary & label and summary the statistics
"""

import argparse
from utils import utils


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='./data/toy', help="Directory containing the dataset")
parser.add_argument('--min_word_freq', default=1, help="The minimum frequency of a word")


if __name__ == '__main__':
  args = parser.parse_args()
  utils.build_ner_profile(args.data_dir, min_word_freq=args.min_word_freq)
