from decision_tree import DecisionTree
from random import choices, choice
from math import sqrt


class RandomForest():

    def __init__(self, training_data, tree_max_depth, tree_min_size, amount, cat_index):
        self.trees = []
        self.training_data = training_data
        self.tree_max_depth = tree_max_depth
        self.tree_min_size = tree_min_size
        self.tree_amount = amount
        self.cat_index = cat_index
        self.train_forest()

    def train_forest(self):
        amount = int(sqrt(len(self.training_data[0])))
        indexes = [i for i in range(len(self.training_data[0]))]
        indexes.remove(self.cat_index)
        for i in range(self.tree_amount):
            tree_data = choices(self.training_data, k=self.tree_amount)
            tree_indexes = self.get_indexes(indexes, amount)
            tree = DecisionTree(tree_data, self.tree_max_depth, self.tree_min_size, tree_indexes, self.cat_index)
            self.trees.append(tree)

    def get_indexes(self, indexes, amount):
        to_ret = []
        for i in range(amount):
            chosen = choice(indexes)
            to_ret.append(chosen)
            indexes.remove(chosen)
        for item in to_ret:
            indexes.append(item)
        return to_ret

    def classify_item(self, row):
        decisions = []
        for tree in self.trees:
            decisions.append(tree.classify(row))
        return max(set(decisions), key=decisions.count)
