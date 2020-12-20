import itertools
import numpy as np
import os

from timeit import default_timer as timer


def get_support(items, dataset):
    """ Gets support of an item in a dataset with a horizontal layout.

    :param items: set of items
    :param dataset: list of transactions
    :return: int
    """
    support_set = [items.issubset(x) for x in dataset]

    return support_set.count(True) / len(dataset)


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
    C = set()
    k = len(next(iter(k_minus_one_itemsets)))

    # merge all sets in k_minus_one_itemsets and add those that have length k
    for p in k_minus_one_itemsets:
        for q in k_minus_one_itemsets:
            if len(p.union(q)) == k + 1:
                C.add(p.union(q))

    # pruning
    to_remove = []
    for itemset in C:
        # get k-1 subsets
        k_minus_one_subsets = get_subsets(itemset, k)
        # check if k-1 subsets in k-1 fis
        if not sets_in_itemsets(k_minus_one_subsets, k_minus_one_itemsets):
            to_remove.append(itemset)
    # remove items
    for tr in to_remove:
        C.remove(tr)

    return C


def apriori(items, dataset, minsupp):
    """ Implementation of the a priori algorithm.

    :param items: list of items
    :param dataset: np array with transactions/itemsets
    :param minsupp: min support
    :return: frequent item sets
    """
    # note: 1-itemsets are in Lk[0], 2-itemsets are in Lk[1], etc.
    L = [set(frozenset([item]) for item in items if get_support_apriori({item}, dataset) >= minsupp)]
    k = 1

    # while we can still generate frequent itemsets
    while len(L[k-1]) > 0:
        # get candidates for k-itemset from k-1 itemset
        Ck = apriori_gen(L[k-1])
        fis_k = set()
        # pruning to get frequent itemsets from Ck
        for candidate in Ck:
            if get_support(candidate, dataset) >= minsupp:
                fis_k.add(candidate)
        # add frequent k-itemset to Lk
        L.append(fis_k)
        k += 1

    return set.union(*L)


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

    Data structure from a priori implementation can't be used since it was implemented with sets of sets.
    Prefixes can't be obtained from frozenset since order is not changeable.

    :param items: list of items
    :param dataset: np array with transactions
    :param minsupp: min support
    :return: frequent item sets
    """
    item_dict = get_vertical_layout(items, dataset)

    L1 = {k: v for k, v in item_dict.items() if len(v) / len(dataset) >= minsupp}
    k = 1

    L = L1
    Lk = []

    while L.keys():
        Lk.append(L)
        Lnext = {}
        # iterate over all combinations of the items from k-1 itemsets with one more item
        for combination in itertools.combinations(L.keys(), 2):
            if k > 1:
                # check if same prefix with length of at least k, if not -> skip
                prefix_length = len(os.path.commonprefix(combination).split(" "))
                if prefix_length < k-1:
                    continue

            # create new key
            key = " ".join(sorted(set([y for x in combination for y in x.split(" ")])))
            Lnext[key] = item_dict[combination[0]].intersection(*[item_dict[x] for x in combination[1:]])

        # update L to current frequent itemsets
        L = {k: v for k, v in Lnext.items() if len(v) / len(dataset) >= minsupp}
        item_dict = Lnext
        k += 1

    return Lk


def read_file(file_path):
    """ There are problems with pd.read_csv() for the retail.csv that's why this implementation.

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
    for f in fis:
        print('{}'.format(f))
    print('Found {} frequent itemsets'.format(len(fis)))
    print('Done after {}s!'.format(end - start))


def run_eclat(items, dataset, minsupp):
    print('Starting ECLAT algorithm...')
    start = timer()
    fis = eclat(items, dataset, minsupp=minsupp)
    end = timer()
    for f in fis:
        print('{}'.format(f))
    lenfis = 0
    for f in fis:
        lenfis += len(f)
    print('Found {} frequent itemsets'.format(lenfis))
    print('Done after {}s!'.format(end - start))


def main():
    # file_path = 'retail.tsv'
    # minsupp = 0.1
    # results: ECLAT: 257.1290893s
    #       A PRIORI: 610.4664944000001s

    file_path = 'items.tsv'
    minsupp = 0.7
    # results: ECLAT: 0.0005955000000000127s
    #       A PRIORI: 0.0009656999999999999s

    dataset = read_file(file_path)
    items = np.unique([item for sublist in dataset for item in sublist])
    items = items[items != '']

    run_eclat(items, dataset, minsupp)
    run_apriori(items, dataset, minsupp)


if __name__ == '__main__':
    main()
