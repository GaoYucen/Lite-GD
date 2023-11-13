# -*- coding: utf-8 -*-
import argparse

# def str2bool(v):
#     return v.lower() in ('true', '1')

parser = argparse.ArgumentParser(description="Pytorch implementation of Pointer-Net")

# System
parser.add_argument('--sys', default='win', type=str, help='System')
# Data
parser.add_argument('--train_size', default=1000, type=int, help='Training data size')
parser.add_argument('--val_size', default=10000, type=int, help='Validation data size')
parser.add_argument('--test_size', default=100, type=int, help='Test data size')
parser.add_argument('--batch_size', default=256, type=int, help='Batch size')
# Train
parser.add_argument('--nof_epoch', default=1000, type=int, help='Number of epochs')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
# GPU
parser.add_argument('--gpu', default=True, action='store_true', help='Enable gpu')
# TSP
parser.add_argument('--nof_points', type=int, default=5, help='Number of points in TSP')
# Network
parser.add_argument('--embedding_size', type=int, default=128, help='Embedding size')
parser.add_argument('--hiddens', type=int, default=512, help='Number of hidden units')
parser.add_argument('--nof_lstms', type=int, default=2, help='Number of LSTM layers')
parser.add_argument('--dropout', type=float, default=0., help='Dropout value')
parser.add_argument('--bidir', default=True, action='store_true', help='Bidirectional')
parser.add_argument('--input_dimen', type=int, default=23, help='Number of input dimension')
parser.add_argument('--seq_len', type=int, default=5, help='Length of sequence')
parser.add_argument('--can_len', type=int, default=21, help='Length of can')
# Train and Test mode
parser.add_argument('--test_flag', action='store_true', help='Test mode')

def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed
    # params = parser.parse_args()
    # return params

# def print_config():
#     config, _ = get_config()
#     print('\n')
#     print('Data Config:')
#     print('* Batch size:', config.batch_size)
#     print('* Sequence length:', config.max_length)
#     print('* Task coordinates:', config.input_dimension)
#     print('\n')
#     print('Network Config:')
#     print('* Restored model:', config.restore_model)
#     print('* Actor input embedding:', config.input_embed)
#     print('* Actor hidden_dim (num neurons):', config.hidden_dim)
#     print('* Actor tan clipping:', config.C)
#     print('\n')
#     if config.training_mode:
#         print('Training Config:')
#         print('* iteration:', config.iteration)
#         print('* Temperature:', config.temperature)
#         print('* Actor learning rate (init,decay_step,decay_rate):', config.lr1_start, config.lr1_decay_step,
#               config.lr1_decay_rate)
#     else:
#         print('Testing Config:')
#     print('\n')
