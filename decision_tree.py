

class DecisionTree():

    def __init__(self, training_data, max_depth, min_size, indexes, cat_index):
        self.training_data = training_data
        self.max_depth = max_depth
        self.min_size = min_size
        self.indexes = indexes
        self.cat_index = cat_index
        self.root = None
        self.create_tree()

    def gini_index(self, groups, categories):
        amount_of_elem = float(sum([len(group) for group in groups]))
        gini = 0.0
        for group in groups:
            size = float(len(group))
            if size == 0:
                continue
            score = 0.0
            for category in categories:
                prop = [row[self.cat_index] for row in group].count(category) / size
                score += prop**2
            gini += (1.0 - score) * (size / amount_of_elem)
        return gini

    def split_by_value(self, index, value, data_set):
        left, right = list(), list()
        for row in data_set:
            if row[index] < value:
                left.append(row)
            else:
                right.append(row)
        return left, right

    def get_split(self, data_set):
        categories = list(set(row[self.cat_index] for row in data_set))
        best_index, best_value, best_score, best_groups = 0, 0, 99, None
        for index in self.indexes:
            for row in data_set:
                groups = self.split_by_value(index, row[index], data_set)
                gini = self.gini_index(groups, categories)
                if gini < best_score:
                    best_index, best_value, best_score, best_groups = index, row[index], gini, groups
        return {'index': best_index, 'value': best_value, 'groups': best_groups}

    def terminal_node_value(self, group):
        outcomes = [row[self.cat_index] for row in group]
        return max(set(outcomes), key=outcomes.count)
    
    def same_class(self, group):
        category = group[0][self.cat_index]
        for row in group:
            if row[self.cat_index] != category:
                return -1
        return category

    def insert_node(self, node, depth):
        left, right = node['groups']
        del(node['groups'])
        if not left or not right:
            node['left'] = node['right'] = self.terminal_node_value(left + right)
            return
        if depth >= self.max_depth:
            node['left'], node['right'] = self.terminal_node_value(left), self.terminal_node_value(right)
            return

        left_same_class = self.same_class(left)
        right_same_class = self.same_class(right)

        if left_same_class != -1:
            node['left'] = left_same_class
        elif len(left) <= self.min_size:
            node['left'] = self.terminal_node_value(left)
        else:
            node['left'] = self.get_split(left)
            self.insert_node(node['left'], depth+1)

        if right_same_class != -1:
            node['right'] = right_same_class
        elif len(right) <= self.min_size:
            node['right'] = self.terminal_node_value(right)
        else:
            node['right'] = self.get_split(right)
            self.insert_node(node['right'], depth+1)

    def create_tree(self):
        self.root = self.get_split(self.training_data)
        self.insert_node(self.root, 1)

    def _classify(self, node, row):
        if row[node['index']] < node['value']:
            if isinstance(node['left'], dict):
                return self._classify(node['left'], row)
            else:
                return node['left']
        else:
            if isinstance(node['right'], dict):
                return self._classify(node['right'], row)
            else:
                return node['right']

    def classify(self, row):
        return self._classify(self.root, row)

    def _print_tree(self, node, depth=0):
        if isinstance(node, dict):
            print(f"{' ' * depth} [{node['index']} < {node['value']}]")
            self._print_tree(node['left'], depth+1)
            self._print_tree(node['right'], depth+1)
        else:
            print(f"{' ' * depth}[{node}]")

    def print_tree(self):
        self._print_tree(self.root)
