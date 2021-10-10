#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: Jinfen Li
import numpy


class Performance(object):
    def __init__(self, percision, recall, hit_num):
        self.percision = percision
        self.recall = recall
        self.hit_num = hit_num



class Metrics(object):
    def __init__(self):
        """ Initialization

        :type levels: list of string
        :param levels: eval levels, the possible values are only
                       'span','nuclearity','relation'
        """
        # self.levels = levels
        self.span_perf = Performance([], [], 0)
        self.nuc_perf = Performance([], [], 0)
        self.rela_perf = Performance([], [], 0)
        self.span_num = 0
        self.hit_num_each_relation = {}
        self.pred_num_each_relation = {}
        self.gold_num_each_relation = {}

        self.las_perf = Performance([], [], 0)


    def eval(self, pred, gold):
        """ Evaluation performance on one pair of RST trees

        :type goldtree: RSTTree class
        :param goldtree: gold RST tree

        :type predtree: RSTTree class
        :param predtree: RST tree from the parsing algorithm
        """
        goldbrackets = gold
        predbrackets = pred
        self.span_num += len(goldbrackets)
        self._eval(goldbrackets, predbrackets, idx=1)
        self._eval(goldbrackets, predbrackets, idx=2)
        self._eval(goldbrackets, predbrackets, idx=3)



    def _eval(self, goldbrackets, predbrackets, idx):

        if idx == 1 or idx == 2:
            goldspan = [item[:idx] for item in goldbrackets]
            predspan = [item[:idx] for item in predbrackets]
        elif idx == 3:
            goldspan = [(item[0], item[2]) for item in goldbrackets]
            predspan = [(item[0], item[2]) for item in predbrackets]
        else:
            raise ValueError('Undefined idx for evaluation')
        hitspan = [span for span in goldspan if span in predspan]
        p, r = 0.0, 0.0
        for span in hitspan:
            if span in goldspan:
                p += 1.0
            if span in predspan:
                r += 1.0
        if idx == 1:
            self.span_perf.hit_num += p
        elif idx == 2:
            self.nuc_perf.hit_num += p
        elif idx == 3:
            self.rela_perf.hit_num += p
        p /= len(goldspan)
        r /= len(predspan)
        if idx == 1:
            self.span_perf.percision.append(p)
            self.span_perf.recall.append(r)
        elif idx == 2:
            self.nuc_perf.percision.append(p)
            self.nuc_perf.recall.append(r)
        elif idx == 3:
            self.rela_perf.percision.append(p)
            self.rela_perf.recall.append(r)
        if idx == 3:
            for span in hitspan:
                relation = span[-1]
                if relation in self.hit_num_each_relation:
                    self.hit_num_each_relation[relation] += 1
                else:
                    self.hit_num_each_relation[relation] = 1
            for span in goldspan:
                relation = span[-1]
                if relation in self.gold_num_each_relation:
                    self.gold_num_each_relation[relation] += 1
                else:
                    self.gold_num_each_relation[relation] = 1
            for span in predspan:
                relation = span[-1]
                if relation in self.pred_num_each_relation:
                    self.pred_num_each_relation[relation] += 1
                else:
                    self.pred_num_each_relation[relation] = 1


    def report(self):
        """ Compute the F1 score for different eval levels
            and print it out
        """

        p = numpy.array(self.span_perf.percision).mean()
        r = numpy.array(self.span_perf.recall).mean()
        span_f = (2 * p * r) / (p + r)

        print('F1 score on span level is {0:.4f}'.format(span_f))
        p = numpy.array(self.nuc_perf.percision).mean()
        r = numpy.array(self.nuc_perf.recall).mean()
        nuc_f = (2 * p * r) / (p + r)

        print('F1 score on nuclearity level is {0:.4f}'.format(nuc_f))
        p = numpy.array(self.rela_perf.percision).mean()
        r = numpy.array(self.rela_perf.recall).mean()
        if p+r == 0:
            relation_f = 0
        else:
            relation_f = 2 * p * r / (p + r)
        print('F1 score on relation level is {0:.4f}'.format(relation_f))



        sorted_relations = sorted(self.gold_num_each_relation.keys())
        for relation in sorted_relations:
            hit_num = self.hit_num_each_relation[relation] if relation in self.hit_num_each_relation else 0
            gold_num = self.gold_num_each_relation[relation]
            pred_num = self.pred_num_each_relation[relation] if relation in self.pred_num_each_relation else 0
            precision = hit_num / pred_num if pred_num > 0 else 0
            recall = hit_num / gold_num
            try:
                f1 = 2 * precision * recall / (precision + recall)
            except ZeroDivisionError:
                f1 = 0
            print(
                'Relation\t{:20}\tgold_num\t{:4d}\tprecision\t{:05.4f}\trecall\t{:05.4f}\tf1\t{:05.4f}'.format(relation,
                                                                                                               gold_num,
                                                                                                               precision,
                                                                                                               recall,
                                                                                                               f1))
        return span_f, nuc_f, relation_f

