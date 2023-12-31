# Import packages
import os
from GraphSurgery import GraphSurgeryClassifier
from id4ip import ID4IP
import matplotlib.pyplot as plt
import pyAgrum as gum
import numpy as np
from GraphModel import * 
import pyAgrum.lib.image as gumimage
import pandas as pd
from sklearn.model_selection import train_test_split
import func_timeout
from utils import *
import matplotlib
font = {'weight' : 'bold',
        'size'   : 20}
matplotlib.rc('font', **font)

def run_function(f, list_of_arguments, max_wait, default_value):
    """
        run a function with time restriction on the program. It will exit out of the function once the subprocess is finished and the time is up.

        Arg:
            f: function
            list_of_arguments (list): a list of arguments that should be used by the function f
            max_wait (int): an integer to determine the time restrcition
            default_vaue (None): when nothing is returned, we return None
    """
    try:
        return func_timeout.func_timeout(max_wait, f, args=list_of_arguments)
    except func_timeout.FunctionTimedOut:
        pass
    return default_value

def splitSamplesToXY(graphModel,
                     path_for_sample, 
                     path_for_covariates, 
                     path_for_target,
                     seed,
                     isTestSample=False,
                     saveFiles=True):
    """
        take the sample generated by Pyagrum and split them into features and target

        Arg:
            graphModel(GraphModel object): an object generated by the GraphModel class 
            path_for_sample (str): a filepath to get the training/test file
            path_for_covariates (str): a filepath to save the feature columns dataframe
            path_for_target (str): a filepath to save the target column
            isTestSample (bool): determine whether we are splitting the test sample
            saveFiles (bool): determine whether we need to save the feature columns and targets as files

    """
    X_col = graphModel.graphnames[:]
    X_col.remove(graphModel.target)
    Y_col = [graphModel.target]
    # ready the training data with respect to ith graph
    df = pd.read_csv(path_for_sample)
    X  = df[X_col] # get the features
    y = df[Y_col] # get the target
    X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size=0.3, random_state=seed)
    if isTestSample:
        if saveFiles:
            X.to_csv(path_for_covariates)
            y.to_csv(path_for_target)
        X_all_var_including_U = df.drop(Y_col, axis=1)
        return X, y, X_all_var_including_U
    else:
        if saveFiles:
            X_train.to_csv(path_for_covariates)
            y_train.to_csv(path_for_target)
        return X_train, y_train, X_test, y_test 



def computeLoss(g, estimator_tuple, Xtrain, ytrain, Xtest, ytest, time_restriction):
    """
        compute the loss for a given estimator

        Arg:
            g (obj): an object from GraphModel class
            estimator_tuple (tuple): a tuple that contains a string and an estimator class
            Xtrain (df): a dataframe that contains the features for training
            ytrain (df): a dataframe that contains only the target for training
            Xtest (df): a dataframe that contains the features for testing
            ytest (df): a dataframe that contains only the target for testing
            time_restriction (int): an integer that indicates the time restriction in seconds
    """
    estimator_name, estimator = estimator_tuple
    run_function(estimator.fit, [Xtrain,  ytrain], time_restriction, None)
    predicted_output_filepath = g.parent_directory +  str(estimator_name) + "pred_n_" + str(g.num_nodes) + "_lat_" + str(g.num_latent) + ".csv"
    pred = estimator.predict(Xtest, filepath = predicted_output_filepath)
    loss = zero_one_loss(ytest, pred)
    return loss

def plot_result(result_df, training_sample_size, n , gt_loss, ylabel, filepath=None, savefig=True):
    """
        plot the result for the result of this experiment
        Arg:
            result_df (df): a dataframe that records all the training losses and test losses from each estimator class
            num_node (int):  the number of variables on the graph that is generated for the experiment
            time_restriction (int): number of seconds for time restriction 
            gt_loss (float): a zero and one loss computed for P(target|Pa(target)) 
            filepath (str): a filepath to save the output image 
            savefig (bool): determine whether an image file should be saved
    """
    fig, ax = plt.subplots()
    # get the df with respect to each algorithm
    gse_df = result_df[result_df["Algorithm"]=="GSE"]
    id4ip_df = result_df[result_df["Algorithm"]=="ID4IP"]
    # plot GSE result
    plt.errorbar(gse_df["training_sample_size"].tolist(), gse_df["mean"].tolist(), gse_df["se"].tolist() ,marker = "h", linestyle = "solid", color = "m", label="n=" +str(n)+", GSE")
    # plot id4ip 
    plt.errorbar(id4ip_df["training_sample_size"].tolist(), id4ip_df["mean"].tolist(), id4ip_df["se"].tolist(),marker = "*", linestyle= "dashed",color = "c", label="n=" +str(n)+", ID4IP")
    # axis labels
    plt.xlabel('Training Sample Size', fontweight = "bold")
    plt.ylabel(ylabel, fontweight = "bold")
    plt.ylim(0, 0.51)
    if gt_loss is not None:
        plt.axhline(y=gt_loss, color='green', label="P(Y|Pa(Y))")
    # add red color 
    plt.axhline(y=0.5, color='red', label="Worse Case")
    # show the legend
    plt.legend(loc='lower left', prop={'size': 9})
    plt.gcf().subplots_adjust(bottom=0.20)
    if savefig:
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()


def getGroundTruthLoss(Xtest_with_U, ytest, target, idToName, bn, loss_func):
    """
        compute the loss for P(target|Pa(target))

        Arg:
            Xtest_with_U (df): a dataframe that represents testing data, including the latent variables in the columns 
            ytest (df): 
            target (str): a target name in string
            idToName (dict): a dictionary that maps variable ID from the PyAgrum BayesNet object to their string names
            bn (PyArgum BayesNet object): an object generated by PyAgrum BayesNet class
            loss_func (function): a loss function
    """
    parents_ids = bn.parents(target)
    # get all names of the target's parents including the latents
    conditioning_set = [idToName[id] for id in parents_ids] # convert id to string names
    # get a list of all unique zero and one combinations that appear in the test set
    ls_of_combination = binary_combinations(len(conditioning_set))
    Xtest_with_U_copy = Xtest_with_U.copy()
    Xtest_with_U_copy.loc[:,target] = None
    # loop through each combination
    for counter, i in enumerate(ls_of_combination):
        evidence = dict(zip(conditioning_set, i))
        # compute the posterior based on the sample 
        p = gum.getPosterior(bn, evs= evidence, target=target)
        # get the correspond value according to the max
        pred_val = p.argmax()[0][0][target]
        mask = (Xtest_with_U_copy[conditioning_set]==i).all(axis=1)
        # set the predicted value based on the p(y|pa(y))
        Xtest_with_U_copy.loc[mask, target] = pred_val
    pred = Xtest_with_U_copy[[target]]
    # evlauate the loss
    test_loss = loss_func(ytest, pred)
    return test_loss


def eval(estimators, Xtrain,  ytrain, Xtest, ytest, loss_func):
    """
        a pipeline that takes all the queries updated at each time instance and compile them into a dataframe
        Arg:
            estimators (tuple) : a tuple that contains a string and an estimator class
            Xtrain (df): a dataframe that contains the features for training
            ytrain (df): a dataframe that contains only the target for training
            Xtest (df): a dataframe that contains the features for testing
            ytest (df): a dataframe that contains only the target for testing
            loss_func (function): a loss function
    """
    dfs = []
    for estimator_tuple in estimators:
        estimator_name, estimator = estimator_tuple
        print("Fitting {}...".format(estimator_name))
        estimator.fit(Xtrain,  ytrain, loss_func, config["time_restriction"])
        print("----Finished------")
        # get the test losses from all causal queries found for the estimator
        test_losses = []
        time_instances = []
        if not estimator.id_results:
            # if the estimator returns an empty list, we treat that as 0.5 test loss
            test_losses.append(0.5)
            estimator.train_losses.append(0.5)
            time_instances.append(config["time_restriction"])
        else:
            for id_result in estimator.id_results:
                cpt = id_result.eval()
                y_pred = estimator.predict(Xtest, cpt)
                test_loss = loss_func(ytest, y_pred)
                test_losses.append(test_loss)
            time_instances = estimator.time_instances
            # Ensure the length of time instances and the losses are the same
        if len(test_losses) != len(estimator.time_instances):
            # we pad the losses
            length_diff = len(estimator.time_instances) - len(test_losses)
            none_ls =  [None for _ in range(length_diff)]
            test_losses =  test_losses + none_ls

        df = pd.DataFrame({"Runtime, sec": time_instances, "Test Loss": test_losses, "Train Loss": estimator.train_losses})
        df["Algorithm"] = estimator_name
        dfs.append(df)
    # concate all dfs 
    return pd.concat(dfs)



def main(**config):
    """
        run the pipeline of the experiment
    """
    path = config["parent_directory"]
    # Check whether the specified path exists or not
    isExist = os.path.exists(path)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(path)
        print("The new directory is created!")
    list_of_result_dfs = []
    graph_idx = 0
    graph_skip = 0
    while graph_idx < config["num_graphs"]:
        config["graphseed"] = graph_idx + graph_skip + 600
        # instantiate a graph model 
        repeatthesamegraph = False
        try:
            for sampling_idx in range(config["num_sampling"]):
                # generate training sample
                g = GraphModel(**config)
                g.sampleseed = 2000 + sampling_idx
                train_data_output_filepath = g.parent_directory + "train_sampleFromBN_n" + str(g.num_nodes) + "_graph_"+ str(graph_idx) +"_sampling_" + str(sampling_idx) +".csv"
                g.generateTrainingSample(config["usePerfectDist"], train_data_output_filepath, repeatthesamegraph= repeatthesamegraph)
                
                if g.bn_without_latent is None:
                    print("Fail to learn the probability distribution from data. Exit out of the program. Please increase the number of training samples.")
                    return None, None
                print("Picked the target:{}".format(g.target))
                    
                ### training phrase

                # split the training and vadliation data
                Xtrain_output_filepath = g.parent_directory + "X_train_n_" + str(g.num_nodes) +"_graph_"+ str(graph_idx) + "sampling_" + str(sampling_idx) + ".csv"
                ytrain_output_filepath = g.parent_directory + "y_train_n_" + str(g.num_nodes) + "_graph_"+ str(graph_idx) +"sampling_" + str(sampling_idx) +".csv"
                Xtrain, ytrain, Xvalidate, yvalidate = splitSamplesToXY(g,
                                                path_for_sample=train_data_output_filepath, 
                                                path_for_covariates= Xtrain_output_filepath, 
                                                path_for_target=ytrain_output_filepath, 
                                                seed=g.sampleseed ,
                                                isTestSample= False,
                                                saveFiles=config["output_train_test_data"])
                    
                # join the Xtrain and ytrain
                Xtrain[g.target] = ytrain
                Xtrain.to_csv(Xtrain_output_filepath)
                if repeatthesamegraph == False:
                    # if it is the first time to see this graph
                    # construct the bn without u by changing every bidirected edge to directed edge 
                    g.create_bn_without_latent()
                # learn the cpt
                g.learnCPT(Xtrain_output_filepath)

                # Add a selection variable
                g.addSelectionVariable()

                # remove the added directed edges if we added any previously inside create_bn_without_latent()
                if g.added_edges_tuple:
                    g.remove_added_directed_edges_from_bn_without_latent()
                
                # convert bn to causal models
                g.convertBayesNetToCausalModel()

                # testing phrase
                # generate test sample
                if repeatthesamegraph==False:
                    bn_for_generating_test_samples = gum.BayesNet(g.bn_with_latent_as_observed_without_S) # make a copy
                    gum.initRandom(g.sampleseed)
                    for chs in g.chS:
                        bn_for_generating_test_samples.generateCPT(chs) # randomly intervening on ch(S)
                    test_data_output_filepath = g.parent_directory + "test_sampleFromBN_n" + str(g.num_nodes)  + "_training_size_"+ str(g.train_sample_size) + "_graph_"+ str(graph_idx) + "sampling_" + str(sampling_idx) +".csv"
                    gum.initRandom(g.sampleseed)
                    gum.generateSample(bn_for_generating_test_samples, g.test_sample_size, name_out=test_data_output_filepath)

                    # split the testing data after a variable being intervened by S
                    Xtest__output_filepath = g.parent_directory + "X_test_n_" + str(g.num_nodes) + "_training_size_"+ str(g.train_sample_size) + "_graph_"+ str(graph_idx) +"sampling_" + str(sampling_idx) +".csv"
                    ytest_output_filepath = g.parent_directory + "y_test_n_" + str(g.num_nodes) + "_training_size_"+ str(g.train_sample_size) +"_graph_"+ str(graph_idx) +"sampling_" + str(sampling_idx) +".csv"
                    Xtest_without_U, ytest, Xtest_with_U = splitSamplesToXY(g, 
                                                                path_for_sample = test_data_output_filepath,
                                                                path_for_covariates= Xtest__output_filepath, 
                                                                path_for_target=ytest_output_filepath, 
                                                                seed = g.sampleseed,
                                                                isTestSample= True,
                                                                saveFiles=config["output_train_test_data"])
                    
                    Xtest_with_U.to_csv(g.parent_directory + "Xtest_with_U_n_" + str(g.num_nodes) + "_training_size_"+ str(g.train_sample_size) +"_graph_"+ str(graph_idx) +"sampling_" + str(sampling_idx) +".csv")
                    # output graph image
                    if config["output_graph_image"]:
                        image_filepath = g.parent_directory + "graph_n"+ str(g.num_nodes) + "_graph_"+ str(graph_idx) +".png"
                        # generate the image based on causal model that has a selection variable
                        gumimage.export(g.cm_with_S, image_filepath)
                        print("Graph generated!")
                        
                # fit all estimators with training data
                estimators = [
                            ("ID4IP", ID4IP(g)),
                            ("GSE", GraphSurgeryClassifier(g))
                            ]
                    
                result_df = eval(estimators, Xvalidate, yvalidate, Xtest_without_U, ytest, zero_one_loss)

                result_df = result_df.reset_index(drop=True)
                result_df.to_csv(g.parent_directory + "result_df_n_" + str(g.num_nodes) + "_training_size_"+ str(g.train_sample_size) +"_graph_"+ str(graph_idx) +"sampling_" + str(sampling_idx) +".csv")
                result_df = result_df.loc[result_df.reset_index().groupby(["Algorithm"])["Runtime, sec"].idxmax()]
                result_df = result_df.reset_index(drop=True)
                list_of_result_dfs.append(result_df)
                repeatthesamegraph = True
                print("-----Comparison-------")
                print(result_df)
        except Exception as e:
            print("Skipping graph due to the following error!")
            print(e)
            # resample the seed for changing the graph
            graph_skip+=1
            continue
        # increment in case the number of graph > 1
        graph_idx += 1

    
    final_df = pd.concat(list_of_result_dfs)
    # compute the mean and standard deviation
    final_test_df = final_df.groupby(['Algorithm'])['Test Loss'].agg(['mean', 'std', 'count']).reset_index()
    # compute the standard error by dividing std by sqrt of n-1
    final_test_df["se"] = final_test_df['std']/np.sqrt(final_test_df['count']-1)

    # get train loss
    # compute the mean and standard deviation
    final_train_df = final_df.groupby(['Algorithm'])['Train Loss'].agg(['mean', 'std', 'count']).reset_index()
    # compute the standard error by dividing std by sqrt of n-1
    final_train_df["se"] = final_train_df['std']/np.sqrt(final_train_df['count']-1)

    gt_loss = getGroundTruthLoss(Xtest_with_U, ytest, g.target, g.idToName, g.bn_with_latent_as_observed_without_S, zero_one_loss)
    return final_test_df, final_train_df, gt_loss



if __name__ == "__main__":
    # specify the configuration
    n = [16, 25, 32]
    training_sample_size = [50, 100, 250, 500, 1000, 2500, 5000, 10000]
    num_graphs = 1
    num_sampling = 3
    for i in range(len(n)):
        list_of_res_df = []
        list_of_train_df = []
        for current_training_sample_size in training_sample_size:
            print("------TRAINING SAMPLE SIZE : {}----------------------".format(current_training_sample_size))
            num_latents = int(n[i]/2)
            config = {
                "num_graphs": num_graphs,
                "num_sampling": num_sampling,
                "num_nodes": n[i], # number of observed variables
                "num_latent": num_latents, # number of unobserved variables, must be greater than 0
                "time_restriction": 200, # number of seconds to restrict program execution
                'arc_ratio': 3,  
                "parent_directory": "tr_samplesize_vs_loss_results/", # root filepath to save output files
                "train_sample_size": current_training_sample_size,  # number of samples generated from the true DAG 
                "test_sample_size": 10000,
                'varyingTrainingCPT': False,
                "output_train_test_data": True,
                "output_graph_image": True,
                "usePerfectDist": False, # if false, it uses pyagrum to learn the distribution from data
                "user_provided_bayesNet": None, # provide a graph where everything is observed, including the latents you want to specify
                "user_provided_target": None, # # specify the target if "user_provided_graph" is provided
                "user_provided_latent_spec": None # specify the latent specification if "user_provided_graph" is provided
            }

            # The following is an example for using a graph provided by the user 
            # bn = gum.fastBN("Z->X->Y; U->Z; U->Y")
            # latent_spec = [("U", ("Z", "Y"))] 
            #
            # config = {
            #     "num_nodes": 3, 
            #     "num_latent": 1, 
            #     "time_restriction": 600, 
            #     'arc_ratio': 1,   
            #     "parent_directory": "results/", 
            #     "train_sample_size": 100000,  
            #     "test_sample_size": 10000,
            #     "graphseed": 10, # random seed
            #     "sampleseed": 20, # random seed
            #     "output_train_test_data": True,
            #     "output_graph_image": True,
            #     "usePerfectDist": True, 
            #     "user_provided_bayesNet": bn, 
            #     "user_provided_target": "Y", 
            #     "user_provided_latent_spec": latent_spec
            # }

            # call on experiment
            res_df, train_df, _ = main(**config)
            res_df["training_sample_size"] = current_training_sample_size
            train_df["training_sample_size"] = current_training_sample_size
            list_of_res_df.append(res_df)
            list_of_train_df.append(train_df)

        # concat the list of dfs 
        plot_df = pd.concat(list_of_res_df)
        plot_train_df = pd.concat(list_of_train_df)
        training_sample_size_str_list = [str(nod) for nod in training_sample_size]
        tr_sample_str = "_".join(training_sample_size_str_list)
        plot_df.to_csv(config["parent_directory"] + "test_loss_df_n_"+str(n[i])+ "_training_size_" + tr_sample_str+ ".csv")
        plot_train_df.to_csv(config["parent_directory"] + "train_loss_df_n_"+str(n[i])+ "_training_size_" + tr_sample_str+ ".csv")
        image_output_filepath = config["parent_directory"] + "_n_" + str(n[i]) +  "test_loss_by_time_comparison.png"
        plot_result(plot_df, training_sample_size, n[i], gt_loss=None, ylabel = 'Test Loss', filepath=image_output_filepath, savefig=True)
        image_output_filepath = config["parent_directory"] + "_n_" + str(n[i]) +  "train_loss_by_time_comparison.png"
        plot_result(plot_train_df, training_sample_size, n[i], gt_loss=None, ylabel='Training Loss', filepath=image_output_filepath, savefig=True)
