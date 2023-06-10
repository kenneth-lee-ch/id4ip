import numpy as np

def orientBdedges(bn, conf_child):
    # loop through each confounded component
    for i in conf_child.items():
        u, twonodes = i 
        var1,var2 = twonodes
        id1 = bn.idFromName(var1)
        id2 = bn.idFromName(var2)
        # check whether id2 is a parent or child of id1
        id1pa = bn.parents(id1)
        id1ch = bn.children(id1)
        added_edges_tuple = []
        # remove the U from the graph
        bn.erase(u)
        if id2 in id1pa or id2 in id1ch:
            continue
        else:
            # check whether id2 is a descendant of id1
            id1de = bn.descendants(id1)
            if id2 in id1de:
                bn.addArc(id1,id2)
                added_edges_tuple.append((id1, id2))
            else:
                bn.addArc(id2,id1)
                added_edges_tuple.append((id2, id1))
    return bn, added_edges_tuple


    
def zero_one_loss(y, y_pred):
    """
        compute the loss given the predicted labels and true labels

        Arguments:
            y_pred (dataframe): the predicted labels
            y (dataframe): the actual labels
    """
    return np.mean((y_pred - y)**2, axis=0)[0]


def binary_combinations(n):
    """
        Returns a list of all possible combinations of 0 and 1 for a given integer n.
    """
    if n == 0:
        return [[]]
    else:
        prev_combinations = binary_combinations(n - 1)
        return [[0] + comb for comb in prev_combinations] + [[1] + comb for comb in prev_combinations]
    