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
from typing import Mapping, List, Tuple

class ShadowDecTree:
    """
    The decision trees for classifiers and regressors from scikit-learn
    are built for efficiency, not ease of tree walking. This class
    is intended as a way to wrap all of that information in an easy to use
    package.

    This tree shadows a decision tree as constructed by scikit-learn's
    DecisionTree(Regressor|Classifier).  As part of build process, the
    samples considered at each decision node or at each leaf node are
    saved as a big dictionary for use by the nodes.

    Field leaves is list of shadow leaf nodes. Field internal is list of
    shadow non-leaf nodes.

    Field root is the shadow tree root.
    """
    def __init__(self, tree_model, X_train,
                 feature_names : List[str],
                 class_names : (List[str],Mapping[int,str])=None):
        self.tree_model = tree_model
        self.feature_names = feature_names
        self._class_names = class_names
        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.values
        self.X_train = X_train
        self.node_to_samples = ShadowDecTree.node_samples(tree_model, X_train)

        tree = tree_model.tree_
        children_left = tree.children_left
        children_right = tree.children_right

        # use locals not args to walk() for recursion speed in python
        leaves = []
        internal = [] # non-leaf nodes

        def walk(node_id):
            if (children_left[node_id] == -1 and children_right[node_id] == -1):  # leaf
                t = ShadowDecTreeNode(self, node_id)
                leaves.append(t)
                return t
            else:  # decision node
                left = walk(children_left[node_id])
                right = walk(children_right[node_id])
                t = ShadowDecTreeNode(self, node_id, left, right)
                internal.append(t)
                return t

        root_node_id = 0
        # record root to actual shadow nodes
        self.root = walk(root_node_id)
        self.leaves = leaves
        self.internal = internal

    def nclasses(self):
        return self.tree_model.tree_.n_classes[0]

    def nnodes(self) -> int:
        "Return total nodes in the tree"
        return self.tree_model.tree_.node_count

    def leaf_sample_counts(self) -> List[int]:
        return [self.tree_model.tree_.n_node_samples[leaf.id] for leaf in self.leaves]

    def isclassifier(self):
        return self.tree_model.tree_.n_classes > 1

    def class_name(self, target_value : int):
        return self._class_names[target_value] # _class_names is either dict or list

    def class_names(self):
        if isinstance(self._class_names, dict):
            # sort by the class value (not name)
            sorted_by_key = sorted(self._class_names.items(), key=lambda x: x[0])
            return [self._class_names[key] for key in sorted_by_key]
        return [n for n in self._class_names if n is not None]

    @staticmethod
    def node_samples(tree_model, data) -> Mapping[int, list]:
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
    pointers to child nodes, if any.  As part of tree construction process, the
    samples examined at each decision node or at each leaf node are
    saved into field node_samples.
    """
    def __init__(self, shadowtree, id, left=None, right=None):
        self.shadowtree = shadowtree
        self.id = id
        self.left = left
        self.right = right

    def split(self) -> (int,float):
        return self.shadowtree.tree_model.tree_.threshold[self.id]

    def feature(self) -> int:
        return self.shadowtree.tree_model.tree_.feature[self.id]

    def feature_name(self) -> (str,None):
        if self.shadowtree.feature_names is not None:
            return self.shadowtree.feature_names[ self.feature() ]
        return None

    def samples(self) -> List[int]:
        """
        Return a list of sample indexes associated with this node. If this is a
        leaf node, it indicates the samples used to compute the predicted value
        or class.  If this is an internal node, it is the number of samples used
        to compute the split point.
        """
        return self.shadowtree.node_to_samples[self.id]

    def nsamples(self) -> int:
        """
        Return the number of samples associated with this node. If this is a
        leaf node, it indicates the samples used to compute the predicted value
        or class. If this is an internal node, it is the number of samples used
        to compute the split point.
        """
        return self.shadowtree.tree_model.tree_.n_node_samples[self.id] # same as len(self.node_samples)

    def split_samples(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return the list of indexes to the left and the right of the split value.
        """
        samples = np.array(self.samples())
        node_X_data = self.shadowtree.X_train[samples,self.feature()]
        split = self.split()
        left = np.nonzero(node_X_data < split)[0]
        right = np.nonzero(node_X_data >= split)[0]
        return left, right

    def isleaf(self) -> bool:
        return self.left is None and self.right is None

    def isclassifier(self):
        return self.shadowtree.tree_model.tree_.n_classes > 1

    def prediction(self) -> (int,None):
        """
        If this is a leaf node, return the predicted continuous value, if this is a
        regressor, or the class number, if this is a classifier.
        """
        if not self.isleaf(): return None
        if self.isclassifier():
            counts = np.array(tree.value[self.id][0])
            predicted_class = np.argmax(counts)
            return predicted_class
        else:
            return self.shadowtree.tree_model.tree_.value[self.id][0][0]

    def prediction_name(self) -> (str,None):
        """
        If the tree model is a classifier and we know the class names,
        return the class name associated with the prediction for this leaf node.
        """
        if self.isclassifier():
            return self.shadowtree.class_names[ self.prediction() ]
        return None

    def class_counts(self) -> (List[int],None):
        """
        If this tree model is a classifier, return a list with the count
        associated with each class.
        """
        if self.isclassifier():
            return np.array(self.shadowtree.tree_model.tree_.value[self.id][0])
        return None

    def __str__(self):
        if self.left is None and self.right is None:
            return "pred={value},n={n}".format(value=round(self.prediction(),1), n=self.nsamples())
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