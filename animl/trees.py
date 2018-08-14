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
    DecisionTree(Regressor|Classifier).  As part of build process, the
    samples considered at each decision node or at each leaf node are
    saved into a node field called node_samples.

    Field leaves is list of shadow leaf nodes. Field internal is list of
    shadow non-leaf nodes.
    """
    def __init__(self, tree_model, X_train, feature_names=None, class_names=None):
        self.tree_model = tree_model
        self.feature_names = feature_names
        self.class_names = class_names
        tree = tree_model.tree_
        children_left = tree.children_left
        children_right = tree.children_right

        node_to_samples = ShadowDecTree.get_node_samples(tree_model, X_train)

        # use locals not args to walk() for recursion speed in python
        leaves = []
        internal = [] # non-leaf nodes

        def walk(node_id):
            if (children_left[node_id] == -1 and children_right[node_id] == -1):  # leaf
                t = ShadowDecTreeNode(self, node_id, node_samples=node_to_samples[node_id])
                leaves.append(t)
                return t
            else:  # decision node
                left = walk(children_left[node_id])
                right = walk(children_right[node_id])
                t = ShadowDecTreeNode(self, node_id, left, right,
                                      node_samples=node_to_samples[node_id])
                leaves.append(t)
                return t

        root_node_id = 0
        # record root to actual shadow nodes
        self.root = walk(root_node_id)
        self.leaves = leaves
        self.internal = internal

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

    def __str__(self):
        return str(self.root)


class ShadowDecTreeNode:
    """
    A node in a shadow tree.  Each node has left and right
    pointers to child nodes, if any.  As part of build process, the
    samples considered at each decision node or at each leaf node are
    saved into field node_samples.
    """
    def __init__(self, shadowtree, id, left=None, right=None, node_samples=None):
        self.shadowtree = shadowtree
        self.id = id
        self.left = left
        self.right = right
        self.node_samples = node_samples

    def split(self):
        return self.shadowtree.tree_model.tree_.threshold[self.id]

    def feature(self):
        return self.shadowtree.tree_model.tree_.feature[self.id]

    def feature_name(self):
        if self.shadowtree.feature_names is not None:
            return self.shadowtree.feature_names[ self.feature() ]
        return self.feature()

    def num_samples(self):
        return self.shadowtree.tree_model.tree_.n_node_samples[self.id] # same as len(self.node_samples)

    def prediction(self):
        is_classifier = self.shadowtree.tree_model.tree_.n_classes > 1
        if is_classifier:
            counts = np.array(tree.value[self.id][0])
            predicted_class = np.argmax(counts)
            return predicted_class
        else:
            return self.shadowtree.tree_model.tree_.value[self.id][0][0]

    def prediction_name(self):
        is_classifier = self.shadowtree.tree_model.tree_.n_classes > 1
        if is_classifier:
            return self.shadowtree.class_names[ self.prediction() ]
        return None

    def __str__(self):
        if self.left is None and self.right is None:
            return "pred={value},n={n}".format(value=round(self.prediction(),1),n=self.num_samples())
        else:
            return "({f}@{s} {left} {right})".format(f=self.feature_name(),
                                                     s=round(self.split(),1),
                                                     left=self.left if self.left is not None else '',
                                                     right=self.right if self.right is not None else '')


if __name__ == "__main__":
    regr = tree.DecisionTreeRegressor(max_depth=5, random_state=666)
    boston = load_boston()

    X_train = pd.DataFrame(boston.data, columns=boston.feature_names)
    y_train = boston.target

    regr = regr.fit(X_train, y_train)

    dtree = ShadowDecTree(regr, X_train, feature_names=X_train.columns)
    print(dtree)