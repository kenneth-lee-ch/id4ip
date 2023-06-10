import pyAgrum as gum



def get_ccom(confTochild, latent_conf, sub_graph, visited, cur_node):
    """
        Find the C-component of "cur_node" based on the knowledge of how latents were added previously to a graph

        Arguments:
            confTochild (dict): a dictionary that map each confounder name to all of its children as a string to list mapping. 
            latent_conf (dict): a dictionary that maps each variable name to all of its latent confounder as a string to list mapping
            subgraph (dict.value): a list of all variable names in the bn
            visited (list): a list to keep track of variables that have not been visited given the list of variable names of the graph
            cur_node (str): a string that denotes a variable name in the graph and find the C-component that contains that variable
        Return:
            A list of the variable that form a c-component that contains "cur_node"
    """

    # we are adding each c-component node that we are visiting
    visited += [cur_node]
    nbrs=[]
    # look at for current node to see if has any latent confounder. 
    # latent_conf is a dictionary with {variable : its latent confounder}
    for conf in latent_conf[cur_node]:
        # for each latent confounder :conf
        # we get the children of that confounder and append that to a list
        nbrs+= confTochild[conf]

    # loop through all the children of the latent confounder of the "current node"
    for nbr in nbrs:
        if (nbr in sub_graph) and (nbr not in visited):
            get_ccom(confTochild, latent_conf, sub_graph, visited, nbr)  #visited array is being updated since its being passed by reference

    # we send back the updated visited array after each recursion call
    return visited

def getMACS(bn_with_no_S, ac_component, confTochild, latent_conf):
    """
        find-MACS-on-set for a singleton target Y

        Arg:
        bn_wth_no_S: a bayes net object that does not have S
        ac_component (list): a list to represent AC-component
        c_components: all c_components of G
        Return:
            graph: a outcome-rooted C-tree    
    """
    # copy the graph so that we don't change the original object
    bn = gum.BayesNet(bn_with_no_S)
    # # get the variable names as a list of strings from the graph
    graph_var_names = list(bn.names())
    graph_ids = bn.ids(graph_var_names)
    graph_dict = dict(zip(graph_ids, graph_var_names))
    ancestors_of_target_set = []
    # Get all ancestors with respect to ac_component
    for v in ac_component:
        AnY_id = list(bn.ancestors(v))
        ancestors_of_target_set=  ancestors_of_target_set + AnY_id
    # use set to get the unique members
    all_ancestors_names_wrt_ac_component = [graph_dict[i]for i in ancestors_of_target_set]
    # add back the target sets since the ancestors are not inclusive in pyargum
    all_ancestors_names_wrt_ac_component = all_ancestors_names_wrt_ac_component + ac_component
    variables_to_remove = [i for i in graph_var_names if i not in all_ancestors_names_wrt_ac_component]
    if variables_to_remove:
        # if the list is non-empty, we erase the node from the bn one by one
        for j in variables_to_remove:
            bn.erase(j)
        # recurse on the resulting bn and 
        return getMACS(bn, ac_component, confTochild, latent_conf)
    # get the C-component
    all_c_components = []
    for v in ac_component:
        c_comp_v = get_ccom(confTochild, latent_conf, graph_var_names, [] , v)
        all_c_components = all_c_components + c_comp_v
    variables_to_remove_all_com = [i for i in graph_var_names if i not in all_c_components]
    if variables_to_remove_all_com:
        for j in variables_to_remove_all_com:
            bn.erase(j)
        return getMACS(bn, ac_component, confTochild, latent_conf)
    return bn