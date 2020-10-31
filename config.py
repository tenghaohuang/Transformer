import logging

import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # sets device for model and PyTorch tensors

# Model parameters
d_model = 300
epochs = 10000
embedding_size = 300
hidden_size = 1024
data_file = 'data.pkl'
vocab_file = 'vocab.pkl'
n_src_vocab = int(2.2e6)
n_tgt_vocab = int(2.2e6)  # target
maxlen_in = 2000
maxlen_out = 100
# Training parameters
grad_clip = 1.0  # clip gradients at an absolute value of
print_freq = 500  # print training/validation stats  every __ batches
checkpoint = None  # path to checkpoint, None if none

# Data parameters
IGNORE_ID = -1
pad_id = 0
sos_id = 1
eos_id = 2
unk_id = 3
num_train = 90406
num_valid = 7405

train_filename = 'hotpot_train_v1.1.json'
# train_translation_zh_filename = 'data/ai_challenger_translation_train_20170904/translation_train_data_20170904/train.zh'
valid_filename = 'hotpot_dev_fullwiki_v1.json'
# valid_translation_zh_filename = 'data/ai_challenger_translation_validation_20170912/translation_validation_20170912/valid.zh'


def get_logger():
    logger = logging.getLogger()
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] [%(threadName)s] %(name)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


logger = get_logger()
