import pyAgrum as gum
import pyAgrum.causal as csl
import random
from utils import *

MAX_COUNT_FOR_LEARNING_SAMPLING_DIST = 100 



class GraphModel():
    def __init__(self, **spec):
        # set the spec as given
        self.num_nodes = spec.get('num_nodes', 8)
        self.num_latent = spec.get('num_latent', 1)
        self.arc_ratio = spec.get("arc_ratio", 3)
        self.latent_spec = spec.get( "user_provided_latent_spec", None)
        self.target = spec.get("user_provided_target", None)
        self.train_sample_size = spec.get('train_sample_size', 100000)
        self.test_sample_size = spec.get('test_sample_size', 10000)
        self.parent_directory = spec.get('parent_directory', "")
        self.usePerfectDist = spec.get('usePerfectDist', True)
        self.bn_with_latent_as_observed_without_S = spec.get('user_provided_bayesNet', None)
        self.graphseed = spec.get('graphseed', 42)
        self.sampleseed = spec.get('sampleseed', 42)
        self.varyingTrainCPT = spec.get('varyingTrainingCPT', False)
        self.bn_without_latent = None
        self.bn_with_S_and_without_latent = None
        self.latent_conf = None
        self.conf_child = None
        self.idToName = None
        self.chS = None
        self.cm_with_S = None
        self.cm_without_S = None
        self.train_sample_path = None
        self.graphnames = None
        self.added_edges_tuple = None
        self.s_name_list = None
        
    def generateRandomGraphWithLatents(self, graphnames):
        """
            generate a random bayes net and add latents randomly to the bayes net

            Arg:
                graphnames (list): a list of strings that represent the names of the variables in the randomly generated DAG
        """
        gum.initRandom(self.graphseed)
        self.bn_with_latent_as_observed_without_S = gum.randomBN(n=self.num_nodes, names=graphnames, ratio_arc=self.arc_ratio, domain_size = 2)
        self.generateLatentVariables(self.bn_with_latent_as_observed_without_S, self.target, self.num_latent)
        return self
    
    def convertLatentSpec(self, latent_spec):
        """
            convert latent specfication in terms of Pyagrum's format to get the information 
            
            Arg:
                latent_spec (list): a list of tuple that contains the specification of the each latent variable to the observed variables
        """
        latent_conf = {}
        conf_child = {} 
        for tup in latent_spec:
            conf, pair_nodes_tuple = tup
            node1, node2 = pair_nodes_tuple
            conf_child[conf].append(node1)
            conf_child[conf].append(node2)
            latent_conf[node1].append(conf)
            latent_conf[node2].append(conf)
        self.latent_conf = latent_conf
        self.conf_child = conf_child
        return self
    
    def removeLatent(self, bn, conf_child):
        """
            remove the observed confounders from bn based on the information in conf_child dictionary
            
            Arg:
                bn (PyArgum BayesNet object): an object generated by PyAgrum BayesNet class
                conf_child (dict): a dictionary that represents a mapping from confounders to their children 
        """
        bn_no_latent = gum.BayesNet(bn)
        allU = list(conf_child.keys())
        allU = [name for name in allU if name in bn.names()]
        for u_var_name in allU:
            bn_no_latent.erase(u_var_name)
        return bn_no_latent
    
    def generateTrainingSample(self, usePerfectDist, train_output_filepath, repeatthesamegraph=False):
        """
            generate a directed acyclic graph randomly with binary variables

            Arg:
                usePerfectDist (bool): determine whether we learn the conditional table from data
                train_output_filepath: a string to represent a filepath to save the generated training sample
        """
        # create a list of strings to specify the variable names
        graphnames = ["X" + str(i) for i in range(self.num_nodes)] 
        self.graphnames = graphnames
        # randomly pick a target variable
        middle_idx = int(self.num_nodes/2)
        bottom_half_graphname = graphnames[middle_idx:]
        random.seed(self.graphseed)
        self.target = random.choice(bottom_half_graphname) # pick the second to the last as the target
        # fix a seed before generating the graph
        if usePerfectDist:
            if self.bn_with_latent_as_observed_without_S is None:
                # If the user did not provide a graph
                self.generateRandomGraphWithLatents(graphnames)
            else:
                # we copy the user provided bayesnet which we do not add latent
                self.convertLatentSpec(self.latent_spec)
                # Remove the observed confounder specified in the latent spec
                self.removeLatent(self.bn_with_latent_as_observed_without_S, self.conf_child)
        else:
             # replace bn with the one learned from finite training sample
            if self.bn_with_latent_as_observed_without_S is None:
                self.generateRandomGraphWithLatents(graphnames)
            else:
                if repeatthesamegraph == False:
                    self.convertLatentSpec(self.latent_spec)
                    self.removeLatent(self.bn_with_latent_as_observed_without_S, self.conf_child)
        # generate the sample from the DAG that has latents as observed
        gum.initRandom(self.sampleseed)
        gum.generateSample(self.bn_with_latent_as_observed_without_S, self.train_sample_size, name_out = train_output_filepath)
        self.train_sample_path = train_output_filepath
        return self

    def remove_added_directed_edges_from_bn_without_latent(self):
        for id1, id2 in self.added_edges_tuple:
            self.bn_without_latent.eraseArc(id1, id2)

    def create_bn_without_latent(self):
        self.bn_without_latent = gum.BayesNet(self.bn_with_latent_as_observed_without_S) # make a copy to avoid changing the original bn object
        if self.conf_child:
            # we modify the graph when if there exists latent confounders
            self.bn_without_latent, added_edges_tuple = orientBdedges(self.bn_without_latent, self.conf_child)
            self.added_edges_tuple = added_edges_tuple

    def learnCPT(self, train_data_output_filepath):
        # remove all Us before we learn the CPT
        learner = gum.BNLearner(train_data_output_filepath, self.bn_without_latent)
        learner.useSmoothingPrior(1)
        learner.fitParameters(self.bn_without_latent) # this function will change the bn that gets passed into it
 

    def generateLatentVariables(self, bn, target, max_num_latent, confounded_edges=None):
        """
            Take a BayesNet object and add latent varible randomly to the graph

            Arg:
                bn (PyArgum BayesNet): a Bayesian network
                max_num_latent: the number latent variables you want to add the graph
            Return:
                bn (PyArgum BayesNet): a modified Bayesian network that has latent variables
                conf_child (dict): a dictionary that map each confounder name to all of its children as a string to list mapping.  
                latent_conf (dict): a dictionary that maps each variable name to all of its latent confounder as a string to list mapping
                idToname: a dictionary as mapping from id to string name of all varaibles in the bayesnet
        """
        idToname = {}
        latent_conf = {}
        latent_spec_list = []
        confounder_assign_prob= 0.20
        targetId = bn.idFromName(target)
        # Create a id to name dictionary
        for var_name in bn.names():
            idToname[bn.idFromName(var_name)] = var_name
            latent_conf[var_name] = []

        # get ids 
        ls_of_ids = bn.nodes()
        # specify the number of latent confounders with at most 4
        num_conf = min(max_num_latent, len(ls_of_ids))
        conf_child = {} # confounder child
        # a set of (id1,id2) tuple to denote the edges in the graph

        # if there is a provided list of confounded edge, we use them instead
        if confounded_edges!=None:
            edge_list= confounded_edges
            confounder_assign_prob=1
        else:
            edge_list = bn.arcs()
            head_list = []
            tail_list = []
            # Ensure there is a latent variable pointing to the target variable
            for e in edge_list:
                if (e[0]==targetId or e[1]==targetId) and not head_list:
                    head_list.append(e)
                else:
                    tail_list.append(e)
            # reorder the edge list so that the latent connected with target will be added first
            edge_list = head_list + tail_list
        # fix a random seed 
        random.seed(self.graphseed)
        unames_to_remove = []
        # for each in num of latent confounders
        for i in range(num_conf):
            nd = 'U' + str(i) # latent variable name
            conf_child[nd] = [] # confounder variable name: id
            # add the node 
            bn.add(nd, 2)
            # get the id of the varaible after adding the 
            uid = bn.idFromName(nd)
            # add var name to dict for id:name
            idToname[uid] = nd
            
            for e in edge_list:
                # pick a random chance?

                ############ new code starts ############
                st, en = e[0], e[1]
                stn = idToname[st]
                enn = idToname[en]
                conf_prob= confounder_assign_prob

                if stn==target or enn==target:
                    if len(latent_conf[target])==0:
                        conf_prob=1
                    else:
                        conf_prob=0.50

                ############ new code ends ############
                
                if random.uniform(0, 1) > conf_prob :
                    continue


                # get their respective variable names

                
                if stn in conf_child[nd] or enn in conf_child[nd] or (set(latent_conf[stn]) & set(latent_conf[enn]) != set({})):
                    # if any of those varaibles is in the children of the current latent confounder, we don't add and move on to the next latent confounder
                    continue
                # add latent confounders to two variables 
                bn.addArc(uid, st)
                bn.addArc(uid, en)
                # use the dictionary to keep track of which variable is in the children of that confounder
                conf_child[nd].append(stn)
                conf_child[nd].append(enn)
                # keep track of all latent variable names and their respective latent confounder variable name in graph. 
                latent_conf[stn].append(nd)
                latent_conf[enn].append(nd)
                latent_spec_list.append((nd, (stn,enn)))
                break

            # if the children list returns empty, we remove that latent
            if not conf_child[nd]:
                    bn.erase(nd)
                    unames_to_remove.append(nd)
                    idToname.pop(uid, None)
        # make sure strictly positive distribution
        for var_name in idToname.values():
            gum.initRandom(self.graphseed)
            bn.generateCPT(var_name)
        # make a copy of the dag and remove all U's from the dag 
        bn_no_latent = self.removeLatent(bn, conf_child)
        
        # remove any U that has no children 
        for uname in unames_to_remove:
            conf_child.pop(uname) 

        # update the spec
        self.latent_spec = latent_spec_list
        self.idToName = idToname
        self.conf_child = conf_child
        self.latent_conf = latent_conf
        self.bn_without_latent = bn_no_latent
        self.bn_with_latent_as_observed_without_S = bn
        return self

    def addSelectionVariable(self):
        """
            add one selection variable randomly to a variable in the graph

        """
        if self.bn_without_latent is None:
            assert("Error. BayesNet needs to be generated before converting it to causal model!")
        else:
            bn_with_S = gum.BayesNet(self.bn_without_latent)
            # get all the names 
            var_names_besides_S = list(bn_with_S.names())
            random.seed(self.graphseed)
            # randomly pick between 1 and 3 to determine how many S we add to the graph
            picked_num = random.randint(1,3)
            middle_idx = int(self.num_nodes/2)
            upper_half_graphnames = self.graphnames[:middle_idx]
            ls_of_chS = []
            s_name_list = []
            for i in range(picked_num):
                s_str = "S" + str(i)
                # add S
                if len(upper_half_graphnames) > 0:
                    bn_with_S.add(s_str, 2)
                    # randomly pick a target variable
                    # pick a node randomly to be chS
                    chS= random.choice(upper_half_graphnames) 
                    bn_with_S.addArc(s_str , chS) # S points to A
                    s_name_list.append(s_str)
                    upper_half_graphnames.remove(chS)
            self.chS = ls_of_chS # update chS
            self.s_name_list = s_name_list
            self.bn_with_S_and_without_latent = bn_with_S
        return self

    def convertBayesNetToCausalModel(self):
        """
            convert bayes net object to causalModel object in Pyagrum library
        """
        if self.latent_spec is not None:
            # get a model for search in terms of graphical structure with S
            cm_with_S = csl.CausalModel(self.bn_with_S_and_without_latent, self.latent_spec, True)
            # get another model for estimating the probability
            cm_without_S = csl.CausalModel(self.bn_without_latent, self.latent_spec, True)
        else:
            # get a model for search in terms of graphical structure with S
            cm_with_S = csl.CausalModel(self.bn_with_S_and_without_latent)
             # get another model for estimating the probability
            cm_without_S = csl.CausalModel(self.bn_without_latent)
        # The difference between two models is the addition of the selection variable
        # Note that the causal model now has latent variables 
        self.cm_with_S = cm_with_S
        self.cm_without_S = cm_without_S
        return self