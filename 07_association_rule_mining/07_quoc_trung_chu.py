import itertools
import numpy as np

from timeit import default_timer as timer


def get_support(items, dataset):
    """ Gets support of an item in an itemset.

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
    """ Implementation of a priori algorithm.

    :param items: list of items
    :param dataset: np array with transactions/itemsets
    :param minsupp: min support
    :return: association rules
    """
    # note: 1-itemsets are in Lk[0], 2-itemsets are in Lk[1], etc.
    print('Getting 1-itemsets...')
    start = timer()
    Lk = [set(frozenset([item]) for item in items if get_support({item}, dataset) >= minsupp)]
    end = timer()
    print('This took {}s!'.format(end - start))
    k = 1

    # while we can still generate frequent itemsets
    while len(Lk[k-1]) > 0:
        print('Getting {}-itemsets...'.format(k+1))
        # get candidates for k-itemset from k-1 itemset
        Ck = apriori_gen(Lk[k-1])
        fis_k = set()
        # pruning to get frequent itemsets from Ck
        for candidate in Ck:
            if get_support(candidate, dataset) >= minsupp:
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


def main():
    # file_path = 'items.tsv'
    # minsupp = 0.7

    file_path = 'retail.tsv'
    minsupp = 0.1

    dataset = read_file(file_path)
    items = np.unique([item for sublist in dataset for item in sublist])
    items = items[items != '']

    print('Starting')
    start = timer()
    fis = apriori(items, dataset, minsupp=minsupp)
    end = timer()
    for f in fis:
        print('{}'.format(f))
    print('Found {} frequent itemsets'.format(len(fis)))
    print('Done after {}s!'.format(end - start))


if __name__ == '__main__':
    main()
