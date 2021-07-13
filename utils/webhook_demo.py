from knockknock import teams_sender
from configparser import ConfigParser

config_parser = ConfigParser()
config_parser.read('config.ini')

@teams_sender(webhook_url=config_parser['WEBHOOK']['teams'])
def train_your_nicest_model():
    import time
    time.sleep(10)
    return {'loss': 0.9} # Optional return value

if __name__ == '__main__':
  train_your_nicest_model()
