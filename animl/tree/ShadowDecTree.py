import IPython, graphviz, re
from io import StringIO
from IPython.display import Image
import numpy as np
import pandas as pd
import math
from sklearn import tree
from sklearn.datasets import load_boston, load_iris
from collections import defaultdict
import string
import re

class ShadowDecTree:
    """
    A tree that shadows a decision tree as constructed by scikit-learn's
    DecisionTree(Regressor|Classifier).  Each node has left and right
    pointers to child nodes, if any.  As part of build process, the
    samples considered at each decision node or at each leaf node are
    saved into field node_samples.
    """
    def __init__(self, tree, id, left=None, right=None, node_samples=None):
        self.tree = tree
        self.id = id
        self.left = left
        self.right = right
        self.node_samples = node_samples

    def split(self):
        return self.tree.threshold[self.id]

    def feature(self):
        return self.tree.feature[self.id]

    def num_samples(self):
        return self.tree.n_node_samples[self.id] # same as len(self.node_samples)

    def prediction(self):
        is_classifier = self.tree.n_classes > 1
        if is_classifier:
            counts = np.array(tree.value[self.id][0])
            predicted_class = np.argmax(counts)
            return predicted_class
        else:
            return self.tree.value[self.id][0][0]

    def __str__(self):
        if self.left is None and self.right is None:
            return "pred={value},n={n}".format(value=round(self.prediction(),1),n=self.num_samples())
        else:
            return "({f}@{s} {left} {right})".format(f=self.feature(),
                                                     s=round(self.split(),1),
                                                     left=self.left if self.left is not None else '',
                                                     right=self.right if self.right is not None else '')

    @staticmethod
    def get_node_samples(tree_model, data):
        """
        Return dictionary mapping node id to list of sample indexes considered by
        the feature/split decision.
        """
        # Doc say: "Return a node indicator matrix where non zero elements
        #           indicates that the samples goes through the nodes."
        dec_paths = tree_model.decision_path(data)

        # each sample has path taken down tree
        node_to_samples = defaultdict(list)
        for sample_i, dec in enumerate(dec_paths):
            _, nz_nodes = dec.nonzero()
            for node_id in nz_nodes:
                node_to_samples[node_id].append(sample_i)

        return node_to_samples

    @staticmethod
    def from_model(tree_model):
        tree = tree_model.tree_
        children_left = tree.children_left
        children_right = tree.children_right

        node_to_samples = ShadowDecTree.get_node_samples(regr, data)

        def walk(node_id):
            if (children_left[node_id] == -1 and children_right[node_id] == -1):  # leaf
                return ShadowDecTree(tree, node_id, node_samples=node_to_samples[node_id])
            else:  # decision node
                left = walk(children_left[node_id])
                right = walk(children_right[node_id])
                return ShadowDecTree(tree, node_id, left, right,
                                     node_samples=node_to_samples[node_id])

        root_node_id = 0
        return walk(root_node_id)
