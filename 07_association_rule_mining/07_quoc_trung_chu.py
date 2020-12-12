import itertools
import numpy as np

from timeit import default_timer as timer


def get_support_apriori(items, dataset):
    """ Gets support of an item in a dataset with a horizontal layout.

    :param items: set of items
    :param dataset: list of transactions
    :return: int
    """
    support_set = [items.issubset(x) for x in dataset]

    return support_set.count(True) / len(dataset)


def get_support_eclat(items, item_dict, dataset):
    """ Gets support of an item in a dataset with a vertical layout.

    :param items: set of items
    :param item_dict: vertical layout
    :param dataset: list of transactions
    """
    items = list(items)
    transactions = item_dict[items[0]]

    for item in items:
        transactions = transactions.intersection(item_dict[item])

    return len(transactions) / len(dataset)


def get_subsets(itemset, length):
    """ Gets subsets of an itemset with the specified length.

    :param itemset: Set of items
    :param length: Length
    :return: list of sets
    """
    tuple_list = list((itertools.combinations(itemset, length)))
    set_list = []
    for element in tuple_list:
        s = set()
        for t in element:
            s.add(t)
        set_list.append(s)

    return set_list


def sets_in_itemsets(sets, itemsets):
    """ Checks if all sets of a list of sets are in itemsets.

    :param sets: list of sets
    :param itemsets: list of sets
    """
    for s in sets:
        if s not in itemsets:
            return False

    return True


def apriori_gen(k_minus_one_itemsets):
    """ Generate a set of k_itemsets which will be candidates for the next iteration.

    :param k_minus_one_itemsets: set of frequent k-1 itemsets
    """
    Ck = set()
    k = len(next(iter(k_minus_one_itemsets)))

    # merge all sets in k_minus_one_itemsets and add those that have length k
    for p in k_minus_one_itemsets:
        for q in k_minus_one_itemsets:
            if len(p.union(q)) == k + 1:
                Ck.add(p.union(q))

    # pruning
    to_remove = []
    for itemset in Ck:
        # get k-1 subsets
        k_minus_one_subsets = get_subsets(itemset, k)
        # check if k-1 subsets in k-1 fis
        if not sets_in_itemsets(k_minus_one_subsets, k_minus_one_itemsets):
            to_remove.append(itemset)
    # remove items
    for tr in to_remove:
        Ck.remove(tr)

    return Ck


def apriori(items, dataset, minsupp):
    """ Implementation of the a priori algorithm.

    :param items: list of items
    :param dataset: np array with transactions/itemsets
    :param minsupp: min support
    :return: frequent item sets
    """
    # note: 1-itemsets are in Lk[0], 2-itemsets are in Lk[1], etc.
    Lk = [set(frozenset([item]) for item in items if get_support_apriori({item}, dataset) >= minsupp)]
    k = 1

    # while we can still generate frequent itemsets
    while len(Lk[k-1]) > 0:
        # get candidates for k-itemset from k-1 itemset
        Ck = apriori_gen(Lk[k-1])
        fis_k = set()
        # pruning to get frequent itemsets from Ck
        for candidate in Ck:
            if get_support_apriori(candidate, dataset) >= minsupp:
                fis_k.add(candidate)
        # add frequent k-itemset to Lk
        Lk.append(fis_k)
        k += 1

    return set.union(*Lk)


def get_vertical_layout(items, dataset):
    """ Make the initially horizontal layout vertical

    :param items: list of items
    :param dataset: np array with all transactions
    """
    item_dict = {}
    for item in items:
        transactions = set()
        for i in range(len(dataset)):
            if item in dataset[i]:
                transactions.add(i)
        item_dict[item] = transactions

    return item_dict


def eclat(items, dataset, minsupp):
    """ Implementation of the eclat algorithm.

    :param items: list of items
    :param dataset: np array with transactions
    :param minsupp: min support
    :return: frequent item sets
    """
    item_dict = get_vertical_layout(items, dataset)
    to_remove = []
    for item in item_dict.keys():
        if get_support_eclat([item], item_dict, dataset) < minsupp:
            to_remove.append(item)
    for tr in to_remove:
        item_dict.pop(tr)

    Lk = [set(frozenset([item]) for item in item_dict.keys())]
    k = 1

    while len(Lk[k-1]) > 0:
        Ck = apriori_gen(Lk[k - 1])
        fis_k = set()
        # pruning to get frequent itemsets from Ck
        for candidate in Ck:
            if get_support_eclat(candidate, item_dict, dataset) >= minsupp:
                fis_k.add(candidate)
        # add frequent k-itemset to Lk
        Lk.append(fis_k)
        k += 1

    return set.union(*Lk)


def read_file(file_path):
    """ There are problems with pd.read_csv() that's why this implementation.

    :param file_path: path to file
    :return: List of transactions with items
    """
    dataset = []

    with open(file_path, 'r') as f:
        transactions = f.readlines()
        for transaction in transactions:
            items = transaction.split('\n')[0]
            items = items.split(' ')
            dataset.append(items)

    f.close()

    return dataset


def run_apriori(items, dataset, minsupp):
    print('Starting A PRIORI algorithm...')
    start = timer()
    fis = apriori(items, dataset, minsupp=minsupp)
    end = timer()
    # for f in fis:
    #     print('{}'.format(f))
    print('Found {} frequent itemsets'.format(len(fis)))
    print('Done after {}s!'.format(end - start))


def run_eclat(items, dataset, minsupp):
    print('Starting ECLAT algorithm...')
    start = timer()
    fis = eclat(items, dataset, minsupp=minsupp)
    end = timer()
    # for f in fis:
    #     print('{}'.format(f))
    print('Found {} frequent itemsets'.format(len(fis)))
    print('Done after {}s!'.format(end - start))


def main():
    file_path = 'retail.tsv'
    minsupp = 0.1
    # results: ECLAT: 257.1290893s
    #       A PRIORI: 610.4664944000001s

    # file_path = 'items.tsv'
    # minsupp = 0.7
    # results: ECLAT: 0.0005955000000000127s
    #       A PRIORI: 0.0009656999999999999s

    dataset = read_file(file_path)
    items = np.unique([item for sublist in dataset for item in sublist])
    items = items[items != '']

    run_eclat(items, dataset, minsupp)
    run_apriori(items, dataset, minsupp)


if __name__ == '__main__':
    main()
