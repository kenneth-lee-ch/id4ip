
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
from itertools import chain, combinations
import time
import pandas as pd
import pyAgrum.causal as csl
import pyAgrum as gum
from findMACS import getMACS

class ID4IP(BaseEstimator, ClassifierMixin):

    def __init__(self, 
                 graphmodel):
        
        self.cm_with_S = graphmodel.cm_with_S
        self.target = graphmodel.target
        self.cm_without_S = graphmodel.cm_without_S
        self.bn_with_latent_as_observed_without_S = graphmodel.bn_with_latent_as_observed_without_S
        self.bn_without_latent_and_S = self.cm_without_S.observationalBN() 
        self.bn_with_S = self.cm_with_S.observationalBN() 
        self.confTochild = graphmodel.conf_child
        self.latent_conf = graphmodel.latent_conf
        self.SID = [self.cm_with_S.idFromName(s) for s in graphmodel.s_name_list]
        self.yid = self.cm_with_S.idFromName(self.target)
        # get a dictionary converting id to variable names
        self.idToName = self.cm_with_S.names()
        self.nameToId_dict = dict((v,k) for k,v in self.cm_with_S.names().items())
        self.loss_min = np.inf
        self.id_result = None
        self.time_instances = []
        self.id_results = []
        self.train_losses = []
        self.conditioning_set = None
        self.intervention_set = None
        self.T_C_var_names = None
        self.isConditioningSetUpdated = False
        self.cpt = None

    
    def update(self, 
               loss, 
               id_result, 
               conditioning_set, 
               intervention_set,
               cpt, 
               time_instance):
        """
            update the training parameters 
        """
        self.id_result = id_result
        self.loss_min = loss
        self.train_losses.append(loss)
        self.time_instances.append(time_instance)
        self.id_results.append(id_result)
        self.conditioning_set = conditioning_set
        self.intervention_set = intervention_set
        self.isConditioningSetUpdated = True
        self.cpt = cpt
        print("Updated!")
        print("Intervention_set:{}".format(self.intervention_set))
        print("Conditioning_set:{}".format(self.conditioning_set))
        print("Training Loss:{}".format(self.loss_min))
        return self
    
    def get_bi_directed_nbrs(self, latent_conf, target, bolded_T):
        """
            find the bidirected neighbors of the target
        """
        nbrs=[]
        for conf in latent_conf[target]:
            for lb in latent_conf:
                if conf in latent_conf[lb] and lb not in bolded_T:
                    nbrs.append(lb)
        return nbrs

    def getUniqueRowValues(self, df):
        """
        Takes a pandas DataFrame as input and outputs a list of tuples of the unique row values across columns.

        Arg:
            df (dataframe): a pandas dataframe 
        
        Return:
            unique_row_values (list): a list of tuples that contain the unique values by row across columns
        """
        unique_rows = []
        seen_rows = set()
        for row in df.itertuples(index=False):
            if row not in seen_rows:
                unique_rows.append(tuple(row))
                seen_rows.add(row)
        return unique_rows
    
    def greedySearchOverConditioningSet(self, ls_v_minus_y_var_name, X, y, bn, loss_func):
        """
            Apply greedy search on searching for conditioning set that minimizes the training loss
        """
        # get the first predictor 
        ls_of_remaining_variables = ls_v_minus_y_var_name[:]
        final_conditioning_list = []
        first_var = ls_v_minus_y_var_name[0]
        final_conditioning_list.append(first_var)
        current_loss = self.getPosteriorNPredict(X, y, bn, loss_func, [first_var])
        ls_of_remaining_variables.remove(first_var)
        # we want to search through ls_of_remaining_variables to see if we can get a better predictor
        hasAnyRemainingNodes = True
        hasUpdated = False
        dummy_list = ls_of_remaining_variables[:]
        while hasAnyRemainingNodes:
            if hasUpdated:
                # replace the ls_of_remaining_variables if have any updates
                ls_of_remaining_variables = dummy_list[:]
                # set this back to false
                hasUpdated = False

            for variable_name in ls_of_remaining_variables:

                if variable_name == ls_of_remaining_variables[-1]:
                    # if this is the last node, we want to break out of the while loop
                    # otherwise, we use the while loop to change the list and continue the next search
                    hasAnyRemainingNodes = False

                larger_conditioning_set = final_conditioning_list + [variable_name]
                loss = self.getPosteriorNPredict(X, y, bn, loss_func, larger_conditioning_set)
                if loss < current_loss:
                    current_loss = loss
                    final_conditioning_list = larger_conditioning_set[:]
                    self.conditioning_set = final_conditioning_list
                    dummy_list.remove(variable_name)
                    # break out and start another forloop with the newlist
                    hasUpdated = True
                    break    
        return final_conditioning_list

    def getPosteriorNPredict(self, X, y, bn, loss_func, conditioning_set = None):
        """
            compute the loss for P(target|features)

            Arg:
                X(df): a dataframe that represents features
                y (df): sample fo the target
                target (str): a target name in string
                idToName (dict): a dictionary that maps variable ID from the PyAgrum BayesNet object to their string names
                bn (PyArgum BayesNet object): an object generated by PyAgrum BayesNet class
                loss_func (function): a loss function
        """
        def setPredVal(X, conditioning_set, sample_tuple, bn):
            """
                local function to set the prediction value to the dataframe for each sample
            """
            ls_tuple = list(sample_tuple)
            ls_int_sample = [int(k) for k in ls_tuple]
            if len(conditioning_set) == 1:
                ls_tuple = list(sample_tuple)
                evidence = dict(zip(conditioning_set, ls_int_sample))
            else:
                evidence = dict(zip(conditioning_set, ls_int_sample))

            p = gum.getPosterior(bn, evs= evidence, target=self.target)
            # get the correspond value according to the max
            pred_val = p.argmax()[0][0][self.target]
 
            # get row index where the row matches with the sample_tuple with respective to the columns set by "conditioning set"
            mask = (X[conditioning_set]==sample_tuple).all(axis=1)
            X.loc[mask, self.target] = pred_val
            return X 
        # take the columns that refer to the variables in the conditioning set
        # and take each unique row of samples as the ls_of_combination e.g. (0,1), (1,0) ,(1,1), (0,0)
        ls_of_combination = self.getUniqueRowValues(X[conditioning_set])
        # make a copy of the X 
        X_copy = X.copy()
        # assign None to create a new column for the predicted target
        X_copy.loc[:,self.target] = None
        for counter, i in enumerate(ls_of_combination):
            if conditioning_set is None:
                X_copy = setPredVal(X_copy, self.conditioning_set, i, bn)
            else:
                X_copy = setPredVal(X_copy, conditioning_set, i, bn)
        # assign the predicted values to the proxy dataframe
        pred = X_copy[[self.target]]
        # evaluate the loss
        loss = loss_func(y, pred)
        return loss

    def predict(self, 
                X, 
                cpt=None):
        """
            predict 1/0 class based on the given conditional probability table and the given features
        """
        if cpt is None:
            cpt_df = self.cpt.putFirst(self.target).topandas()
        else:
            # convert the format to pandas dataframe
            cpt_df = cpt.putFirst(self.target).topandas()
        pred_list  = []
        # record our prediction for each combinations of discrete values
        for _, val in cpt_df.idxmax(axis=1):
            pred_list.append(int(val)) # select 0 or 1 based on the highest returned probabilitiy in the causal estimand
        X_copy = X.copy()
        X_copy.loc[:, self.target] = None
        multi_index_info = np.transpose(cpt_df).columns
        all_zero_ones_val_combinations = list(multi_index_info)
        features_to_select =  list(multi_index_info.names)
        for counter,i in enumerate(all_zero_ones_val_combinations):
            # get the combination and cast it as a list
            val_to_search = list(i)
            val_to_search = [int(k) for k in val_to_search]
            # filter the X test dataframe
            mask = (X_copy[features_to_select]==val_to_search).all(axis=1)
            # by default, the target column is the last column, we fill in the respective predicted values 
            X_copy.loc[mask, self.target] = pred_list[counter] 
        pred = X_copy[[self.target]]
        return pred
    
    def predict_proba(self, 
                      X ,
                      cpt=None):
        """
            predict (in terms of probability) based on the given conditional probability table and the given features
        """
        if cpt is None:
            cpt_df = self.cpt.putFirst(self.target).topandas()
        else:
            # convert the format to pandas dataframe
            cpt_df = cpt.putFirst(self.target).topandas()

        pred_list = cpt_df[self.target]["1"].tolist() # get the probability of class 1
        X_copy = X.copy()
        X_copy.loc[:, self.target] = None
        multi_index_info = np.transpose(cpt_df).columns
        all_zero_ones_val_combinations = list( multi_index_info)
        features_to_select =  list(multi_index_info.names)
        for counter,i in enumerate(all_zero_ones_val_combinations):
            # get the combination and cast it as a list
            val_to_search = list(i)
            val_to_search = [int(k) for k in val_to_search]
            # filter the X test dataframe
            mask = (X_copy[features_to_select]==val_to_search).all(axis=1)
            # by default, the target column is the last column, we fill in the respective predicted values 
            X_copy.loc[mask, self.target] = pred_list[counter] 
        pred = X_copy[[self.target]]
        return pred
    
    def addBestIP(self, 
                  X_train, 
                  y_train, 
                  T_Y_var_names, 
                  Z, 
                  roots,  
                  starttime,
                  loss_func,
                  getTC=False):
        """
            find the invariant predictor that yields the lowest training loss among all the selected sets
            Arg:
                target(str): the target Y
                X_train (df): a dataframe that contains all the features without U
                y_train (df): a dataframe that only contains the target
                T_Y_var_names (list): a list of string to indicate variable names
                Z (list): a list of strings that indicate the variable names and we need to check the MACS for each of them.
                roots (list): a list that contains target variable name. It is for searching MACs of the AC-component of target and the selected candidate from the list Z.
                getTC (bool): to determine whether we want to get T_C 
        """
        T_J = T_Y_var_names[:] # make sure 
        curly_H = [self.target]
        if getTC:
            T_C_var_names = []
        if Z:
            for H in Z: # each H is a string name
                ac_component = roots + [H]
                T_H = getMACS(self.bn_without_latent_and_S, ac_component, self.confTochild, self.latent_conf)
                T_H_var_names = list(T_H.names())
                if getTC:
                    T_C_var_names = T_C_var_names + T_H_var_names

                isSParent = False    
                # check if S is in the parent of T_H
                for t_H in T_H_var_names:
                    res = list(self.bn_without_latent_and_S.parents(t_H))
                    for sid in self.SID:
                        if sid in list(res):
                            isSParent = True
                        # if that child is Y and S is in the Pa(T_y), we return None, 
                        # None to signify there is no invariant predictor in G
                if not isSParent:
                    curly_H.append(H)
                    # Tj = Tj + T_h_var_names
                    # if some previous c-tree is sub-graph of the current c-tree we ignore the repetitions.
                    T_J+= [tt for tt in T_H_var_names if tt not in T_J]

        # get the parents of T_J
        paT_J_minus_T_J = []
        for i in T_J:
            pa_i = list(self.bn_without_latent_and_S.parents(i))
            paT_J_minus_T_J += [p for p in pa_i if p not in paT_J_minus_T_J]

        paT_J_minus_T_J_var = [self.idToName[id] for id in paT_J_minus_T_J]
        # take out all T_j
        paT_J_minus_T_J_var = [i for i in paT_J_minus_T_J_var if i not in T_J]
        # check whether paT_J_minus_T_J_var is empty
        self.greedy_eval(X_train, 
                         y_train,  
                         paT_J_minus_T_J_var,
                         T_J,
                         curly_H, 
                         loss_func,
                         starttime)
        if getTC:
            self.T_C_var_names = T_C_var_names
        return self
        

    def greedy_eval(self, 
                    X, 
                    y, 
                    intervention_set, 
                    T_J, 
                    curly_H, 
                    loss_func, 
                    starttime):
        """
            apply greedy feature selection for selecting features in ID4IP
        """
        
        conditioning_set = curly_H[:]
        conditioning_set.remove(self.target)
        for _ in range(len(T_J)):
            self.eval(conditioning_set,
                      intervention_set,
                      X,
                      y, 
                      loss_func,
                      starttime)
            self.isConditioningSetUpdated= False
            for tj in T_J:
                if tj not in conditioning_set and tj != self.target:
                    # if the element is not already in A, we add it to make another list
                    larger_conditioning_set = conditioning_set + [tj]
                    # compute the loss
                    self.eval(larger_conditioning_set, 
                            intervention_set, 
                            X, 
                            y, 
                            loss_func,
                            starttime)
            if self.isConditioningSetUpdated:
                conditioning_set = self.conditioning_set
            else:
                return self
        return self

    def eval(self, 
             conditioning_set,
             intervention_set, 
             X, 
             y, 
             loss_func, 
             starttime):
        """
            given the found intervention and conditioning sets, 
            evaluate the causal query and compute the loss
        """
        # take the joint distribution
        joint_target = [self.target] + conditioning_set
        # we avoid using conditioning set for ID to skip the step of converting conditional query to unconditional query
        id_result =  csl.doCalculusWithObservation(self.cm_without_S, 
                                                   on =set(joint_target), 
                                                   doing = set(intervention_set))
        # divide the joint table to get the conditional
        cpt = id_result.eval()
        if conditioning_set:
            evidence = cpt.margSumOut(self.target)
            # get the conditional 
            conditional_int_dist = cpt/evidence
        else:
            conditional_int_dist = cpt
        
        # make sure we dont get P(Y). Since we have only two states, we check that by checking whether the size is 2
        if conditional_int_dist.topandas().size != 2:
            pred = self.predict(X, conditional_int_dist)
            loss_current = loss_func(y, pred)
            if loss_current < self.loss_min:
                self.update(loss_current, 
                            id_result, 
                            conditioning_set,
                            intervention_set, 
                            conditional_int_dist, 
                            time.time() - starttime)
        return self
    
    def fit(self, X, y, loss_func, timelimit=None):
        """
            ID4IP fitting process
        """
        starttime = time.time()
        # 1st phase: Find T_y and Y 
        T_Y = getMACS(self.bn_without_latent_and_S, [self.target], self.confTochild, self.latent_conf)   
        T_Y_var_names = list(T_Y.names())
        # check if S is in Pa(T_y)
        ty_parents_ls = []
        for t_y in T_Y_var_names:
            res = list(self.bn_with_S.parents(t_y))
            ty_parents_ls = ty_parents_ls + res
            for sid in self.SID:
                if sid in list(res):
                    # begin heurstic to greedily search all conditional sets 
                    ls_v_minus_y_var_name = list(self.bn_without_latent_and_S.names())
                    ls_v_minus_y_var_name.remove(self.target)
                    self.greedySearchOverConditioningSet(ls_v_minus_y_var_name, X, y, self.bn_without_latent_and_S, loss_func)
                    raise Exception("ID4IP FAIL. There is no any graph surgery estimator: S is the parent of T_y.")
        if timelimit is not None:
            currentime = time.time() - starttime
            if currentime > timelimit:
                return self
        if ty_parents_ls:
            self.addBestIP(X, 
                           y, 
                           T_Y_var_names, 
                           [], 
                           [], 
                           starttime,
                           loss_func)
        # check if there exists any children
        chY_id = list(self.cm_with_S.children(self.yid))
        chY = [self.idToName[i] for i in chY_id]
        T_C_var_names = []
        if timelimit is not None:
            currentime = time.time() - starttime
            if currentime > timelimit:
                return self
        if chY:
            self.addBestIP(X, 
                           y, 
                           T_Y_var_names, 
                           chY, 
                           [], 
                           starttime,
                           loss_func,
                           getTC=True)
        if T_C_var_names:
            bolded_T = T_Y_var_names + T_C_var_names
        else:
            bolded_T = T_Y_var_names
        bidirected_nbr_of_Y = self.get_bi_directed_nbrs(self.latent_conf, self.target, bolded_T)
        if timelimit is not None:
            currentime = time.time() - starttime
            if currentime > timelimit:
                return self
        if bidirected_nbr_of_Y:
            self.addBestIP(X, 
                           y, 
                           T_Y_var_names, 
                           bidirected_nbr_of_Y, 
                           [self.target], 
                           starttime,
                           loss_func)
        if self.loss_min < np.inf:
            print(f'final query:P({self.target}|do({self.intervention_set}), {self.conditioning_set})')
            print("ID4IP's corresponding training loss: {}".format(self.loss_min))
        else:
            if timelimit is not None:
                currentime = time.time() - starttime
                if currentime > timelimit:
                    return self
            # begin heurstic to greedily search all conditional sets 
            ls_v_minus_y_var_name = list(self.bn_without_latent_and_S.names())
            ls_v_minus_y_var_name.remove(self.target)
            self.greedySearchOverConditioningSet(ls_v_minus_y_var_name, X, y, self.bn_without_latent_and_S, loss_func)
            raise Exception("ID4IP FAIL. There is no any graph surgery estimator after exhaustively searching through possible predictors.")
        return self
