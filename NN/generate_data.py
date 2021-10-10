import os
from utils.tree import RstTree
from collections import defaultdict
import random
from utils.document import Doc

class Example(object):

    def __init__(self, texts, token_dict, label=None):
        self.texts = texts
        self.label = label
        self.token_dict = token_dict


class Gen_Data(object):

    class2rel = {'Attribution':0,
    'Background':1,
    'Cause': 2,
    'Comparison': 3,
    'Condition': 4,
    'Contrast': 5,
    'Elaboration': 6,
    'Enablement': 7,
    'Evaluation': 8,
    'Explanation': 9,
    'Joint': 10,
    'Manner-Means': 11,
    'Topic-Comment': 12,
    'Summary': 13,
    'Temporal': 14,
    'Topic-Change': 15,
    'Textual-Organization': 16,
    'Same-Unit': 17}

    class2form = {'NN':0,'NS':1,'SN':2}

    def __init__(self, train_dir, test_dir):

        self.train_dir = train_dir
        self.test_dir = test_dir
        self.train_data = None
        self.dev_data = None
        self.rst_trees = []
        self.train_examples = []
        self.dev_examples = []
        self.test_examples = []


    def read_rst_trees(self, data_dir=None):
        fdis = [os.path.join(data_dir, fname) for fname in os.listdir(data_dir) if fname.endswith('.dis')]
        fmerges = [os.path.join(data_dir, fname) for fname in os.listdir(data_dir) if fname.endswith('.merge')]
        rst_trees = []
        for fmerge in fmerges:
            doc = Doc()
            texts = {}
            doc.read_from_fmerge(fmerge)
            edu_dict = doc.edu_dict
            token_dict = doc.token_dict
            if len(edu_dict) >= 2:
                for edu_span, tids in edu_dict.items():
                    texts[(edu_span, edu_span)] = {}
                    texts[(edu_span, edu_span)]['text'] = ' '.join([token_dict[t].word for t in tids])
                    texts[(edu_span, edu_span)]['token_ids'] = tids

        for fdi, fmerge in zip(fdis, fmerges):
            print("building tree " + fdi)
            rst_tree = RstTree(fdi, fmerge)
            rst_tree.build()
            rst_trees.append(rst_tree)

        self.rst_trees = rst_trees


    def split_dev(self):
        train = self.rst_trees
        random.seed(1234567890)
        random.shuffle(train)
        split_point = 40
        self.train_data = train
        self.dev_data = train[:split_point]


    def generate_train(self):
        self.read_rst_trees(self.train_dir)
        self.split_dev()
        rst_trees = self.train_data
        depth_counter = []
        for tree in rst_trees:

            dfs_list = tree.dfs_list
            token_dict = tree.doc.token_dict
            edu_dict = tree.doc.edu_dict


            texts = defaultdict(dict)
            gold = {}

            for edu_span, tids in edu_dict.items():
                texts[(edu_span, edu_span)]['text'] = ' '.join([token_dict[t].word for t in tids])
                texts[(edu_span, edu_span)]['token_ids'] = tids

            for node in dfs_list:
                depth_counter.append(node.depth)
                gold[node.edu_span] = (node.form, node.assignRelation, node.parse_break)

            self.train_examples.append(Example(texts, token_dict, gold))


    def generate_dev_test(self, type='test'):
        rst_trees = self.dev_data
        if type == 'test':
            self.read_rst_trees(self.test_dir)
            rst_trees = self.rst_trees
        depth = []

        for tree in rst_trees:

            dfs_list = tree.dfs_list
            token_dict = tree.doc.token_dict
            edu_dict = tree.doc.edu_dict


            texts = defaultdict(dict)
            golds = []
            examples = {}
            for node in dfs_list:
                golds.append((node.edu_span, node.form, node.assignRelation, node.parse_break))
                depth.append(node.depth)
            examples[''] = golds

            for edu_span, tids in edu_dict.items():
                texts[(edu_span, edu_span)]['text'] = ' '.join([token_dict[t].word for t in tids])
                texts[(edu_span, edu_span)]['token_ids'] = tids
            if type == 'test':
                self.test_examples.append(Example(texts, token_dict, golds))
            elif type == 'dev':
                self.dev_examples.append(Example(texts, token_dict, golds))