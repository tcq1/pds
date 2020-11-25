import numpy as np
import pandas as pd

label_attribute = 'tennis'
node_id = 0


class Node:
    def __init__(self, df, attribute_value):
        self.content = df
        self.children = []
        self.split_attribute = None
        self.attribute_value = attribute_value
        global node_id
        node_id += 1
        self.id = node_id

    def set_children(self, children):
        self.children = children

    def set_split_attribute(self, attribute):
        self.split_attribute = attribute

    def get_split_attribute(self):
        return self.split_attribute

    def get_children(self):
        return self.children

    def get_content(self):
        return self.content

    def get_id(self):
        return self.id

    def get_children_ids(self):
        id_list = []
        if len(self.get_children()) > 0:
            for child in self.get_children():
                id_list.append(child.get_id())
        return id_list

    def get_label(self):
        """ Gets majority label of this node
        """
        return self.content[label_attribute].value_counts().index[0]

    def get_attribute_value(self):
        return self.attribute_value

    def only_one_label(self):
        """ If all rows have the same label return True, False otherwise
        """
        return len(self.content[label_attribute].unique()) == 1

    def is_leaf_node(self):
        """ Returns True if node is a leaf node, False otherwise
        """
        return len(self.get_children()) == 0


class DecisionTree:
    def __init__(self, root_node):
        self.root = root_node


def entropy(df):
    """ Returns entropy of a dataset
    """
    ent = 0
    total_length = len(df.index)
    value_counts = df[label_attribute].value_counts()
    for value in value_counts:
        pi = value / total_length
        ent += pi * -np.log2(pi)

    return ent


def information_gain(df, attribute):
    """ Calculates the information gain of an attribute
    """
    ig = 0
    total_length = len(df.index)
    subsets = get_subsets(df, attribute)

    for subset in subsets:
        ig += len(subset.index) / total_length * entropy(subset)

    return entropy(df) - ig


def get_subsets(df, attribute):
    """ Gets subsets of a dataframe with different attribute values of a specific attribute
    """
    attribute_values = df[attribute].value_counts().index
    subsets = []
    for value in attribute_values:
        subset = pd.DataFrame(columns=df.columns)
        for i in range(len(df.index)):
            if df.iloc[i][attribute] == value:
                subset = subset.append(df.iloc[i])
                subset.reset_index(drop=True, inplace=True)

        subsets.append(subset)

    return subsets


def get_best_attribute(df, attributes):
    """ Finds the next best attribute from an attribute list
    """
    return max(attributes, key=lambda attribute: information_gain(df, attribute))


def build_tree(df, attribute_list, attribute_value):
    """ Makes a decision tree
    """
    # create new node
    node = Node(df, attribute_value)

    # if all labels are equal return node
    if node.only_one_label():
        return node

    if len(attribute_list) > 0:
        best_attribute = get_best_attribute(node.get_content(), attribute_list)
        node.set_split_attribute(best_attribute)
        subsets = get_subsets(node.get_content(), best_attribute)
        children = []
        for subset in subsets:
            attribute_list.remove(best_attribute)
            children.append(build_tree(subset, attribute_list, subset[best_attribute][0]))
            attribute_list.append(best_attribute)
        node.set_children(children)

    return node


def print_tree(node):
    """ Traverses through the tree and prints out all nodes with their children, labels and split_attributes
    """

    print("Node ID: {}".format(node.get_id()))
    print("Last split attribute value was {}".format(node.get_attribute_value()))
    print("Children of node: {}".format(node.get_children_ids()))
    print("Next split: {}".format(node.get_split_attribute()))
    print("Node is leaf: {}".format(node.is_leaf_node()))
    print("Label: {}".format(node.get_label()))
    print("")

    for child in node.get_children():
        print_tree(child)


def main():
    file_path = 'data-cls.csv'
    df = pd.read_csv(file_path)
    # get attributes
    attribute_list = df.columns.tolist()
    # remove label attribute
    attribute_list.remove(label_attribute)
    tree = build_tree(df, attribute_list, 'None')
    print_tree(tree)


def test():
    file_path = 'data-cls.csv'
    df = pd.read_csv(file_path)
    attribute_list = df.columns.tolist()
    attribute_list.remove('tennis')
    print(attribute_list)
    subsets1 = get_subsets(df, get_best_attribute(df, attribute_list))
    print(subsets1)


if __name__ == '__main__':
    main()
    # test()
