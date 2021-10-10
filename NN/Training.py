"""
    :author: Jinfen Li
    :url: https://github.com/LiJinfen
"""


import torch.optim as optim
import numpy as np
import torch.nn as nn
from utils import metrics
import torch
from torch.utils.data import Dataset, DataLoader
import copy
from utils import DataHandler


class myDataset(torch.utils.data.Dataset):
    def __init__(self, length):
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return idx


class Train(object):
    def __init__(self, model, batch_size, epoch, lr, lr_decay, weight_decay):

        self.model = model
        self.batch_size = batch_size
        self.epoch = epoch
        self.lr = lr
        self.lr_decay = lr_decay
        self.weight_decay = weight_decay



    def LearningRateAdjust(self, optimizer, epoch, lr_decay_epoch=3):

        if (epoch % lr_decay_epoch == 0) and (epoch != 0):
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * self.lr_decay


    def train(self, train_exps, dev_exps):

        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                               lr=self.lr, betas=(0.9, 0.9), weight_decay=self.weight_decay)



        best_f_rel = 0
        best_f_span = 0
        best_f_nuc = 0
        best_uas = 0
        best_las = 0
        best_model = ''
        best_epoch = 0
        for i in range(self.epoch):
            # print('traing')
            self.LearningRateAdjust(optimizer, i)
            s = myDataset(len(train_exps))
            loader = DataLoader(s, batch_size=self.batch_size, shuffle=True, num_workers=0)
            sample_idx_batches = [i for i in loader.__iter__()]
            # print(sample_idx_batches)

            for batch_id in sample_idx_batches:
                optimizer.zero_grad()
                train_exps_batch = copy.deepcopy(np.array(train_exps)[batch_id.tolist()].tolist())
                # self.model.zero_grad()
                loss_split, loss_form,loss_rel,loss_rel1 = self.model.TrainingLoss(train_exps_batch)
                # print(loss_rel)
                Loss = loss_split/2 +loss_form/3+loss_rel/3+loss_rel1/4
                Loss.backward()
                cur_loss = float(Loss.item())

                print(cur_loss)
                nn.utils.clip_grad_norm_(self.model.parameters(), 5)

                optimizer.step()

            # Convert model to eval
            self.model.eval()
            f_span, f_nuc, f_rel = self.getAccuracy(dev_exps)
            if f_span >= best_f_span:
                best_f_span = f_span
                best_epoch = i
                if f_nuc >= best_f_nuc:
                    best_f_nuc = f_nuc

                if f_rel >= best_f_rel:
                    best_f_rel = f_rel
                best_model = copy.deepcopy(self.model)

            self.model.train()
        return best_model, best_epoch


    def getAccuracy(self, example, best_model=None):

        m = metrics.Metrics()
        for exp in example:
            if best_model:
                pred_batch, depth = best_model.TestingLoss(exp)
            else:
                pred_batch, depth = self.model.TestingLoss(exp)

            golds = exp.label
            new_golds = []

            for g in golds:
                new_golds.extend(DataHandler.get_RelationAndNucleus(g[1],
                            g[2], g[0][0],g[0][1], g[3]))
            m.eval(pred_batch, new_golds)


        span, nuc, relation = m.report()
        return span, nuc, relation
