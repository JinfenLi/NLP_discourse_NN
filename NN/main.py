"""
    author: Jinfen Li
    GitHub: https://github.com/LiJinfen
"""

import torch
import numpy as np
import random
import argparse
from model_sep import ParsingNet
from allennlp.modules.elmo import Elmo
import Training
from generate_data import Gen_Data
import logging
import pickle
import os


def parse_args():
    parser = argparse.ArgumentParser(description='Discourse_NN')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--seed', type=int, default=550, help='Seed number')
    parser.add_argument('--epoch', type=int, default=7, help='Epoch number')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial lr')
    parser.add_argument('--lr_decay', type=int, default=0.8, help='Lr decay epoch')
    parser.add_argument('--weight_decay', type=float, default=0.005, help='Weight decay rate')
    parser.add_argument('--save_path', type=str, default=r'model', help='Model save path')
    args = parser.parse_args()

    return args

if __name__ == '__main__':

    args = parse_args()
    USE_CUDA = torch.cuda.is_available()
    device = torch.device("cuda:0" if USE_CUDA else "cpu")
    weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5"
    options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_options.json"
    elmo = Elmo(options_file=options_file, weight_file=weight_file,
                num_output_representations=3, dropout=0.3, requires_grad=False,do_layer_norm=True).to(device)

    batch_size = args.batch_size
    save_path = args.save_path
    seednumber = args.seed
    epoch = args.epoch
    lr = args.lr
    lr_decay = args.lr_decay
    weight_decay = args.weight_decay

    # Setting random seeds
    torch.manual_seed(seednumber)
    if USE_CUDA:
        torch.cuda.manual_seed_all(seednumber)
    np.random.seed(seednumber)
    random.seed(seednumber)


    generator = Gen_Data(train_dir='../data/TRAINING', test_dir='../data/TEST')
    generator.generate_train()
    train_exps = generator.train_examples
    generator.generate_dev_test(type='dev')
    dev_exps = generator.dev_examples

    generator.generate_dev_test()
    test_exps = generator.test_examples

    model = ParsingNet(elmo, device)
    model = model.to(device)

    # with open('../data/data.pkl', 'wb') as file:
    #     pickle.dump((train_exps, dev_exps, test_exps), file)

    # with open('../data/data.pkl', 'rb') as file:
    #     train_exps, dev_exps, test_exps = pickle.load(file)

    TrainingProcess = Training.Train(model, batch_size, epoch, lr, lr_decay, weight_decay)
    best_model, best_epoch = TrainingProcess.train(train_exps,test_exps)
    torch.save(best_model, os.path.join(args.save_path, r'best.model'))

    print('done training')
    # best_model = torch.load('model/bestall.model')
    TrainingProcess.getAccuracy(test_exps, best_model)
