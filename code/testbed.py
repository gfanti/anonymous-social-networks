#testbed
from spreadparser import Parser
import estimation
import random
from scipy.io import savemat
import sys
import networkx

if __name__ == '__main__':


    pds_opt = []
    pds_opt_nodir = []
    pds_spy = []
    pds_spy_nodir = []
    pds_rand = []
    hops_opt = []
    hops_opt_nodir = []
    hops_spy = []
    hops_spy_nodir = []
    hops_rand = []
    percents_malicious = [5,10,20,30,40,50,60,70,80,90]
    # percents_malicious = [5]
    trials = 2000
    num_nodes = 100

    for percent_malicious in percents_malicious:
        pd_opt = 0
        pd_opt_nodir = 0
        pd_rand = 0
        pd_spy = 0
        pd_spy_nodir = 0
        opt_distances = []
        opt_nodir_distances = []
        entropy_distances = []
        rand_distances = []
        spy_distances = []
        spy_nodir_distances = []
        num_singular = 0
        # percent_malicious = 90
        graph_size = 'N' + str(num_nodes) + '_BA'
        #directory = 'data/' + graph_size + '/malicious_' + str(percent_malicious) + '/'
        for i in range(trials):
            #i += 1000
            if (i % 50) == 0:
                print('Trial ',i)
            #parser = Parser( directory + 'output' + str(i+1))
            parser = Parser(sys.argv[1] + "/infect_" + str(percent_malicious) + '/out' + str(i+1))
            print "Currently testing file", sys.argv[1] + "/infect_" + str(percent_malicious) + '/out' + str(i+1)
            source, adjacency, malicious_nodes, timestamps, infectors = parser.parse_file()
            
            e = estimation.Estimator(adjacency, malicious_nodes, timestamps, infectors)
            
            # print('the diameter is ', e.get_diameter())
            # exit(0)
            
            # Optimal estimator
            opt = estimation.OptimalEstimator(adjacency, malicious_nodes, timestamps, infectors)
            opt_est = opt.estimate_source()
            # print('True source is ',source)
            if opt_est == -1:
                print('NO BUENO')
                num_singular += 1
                continue
            dist_opt = networkx.shortest_path_length(opt.graph,source, opt_est)
            if dist_opt == 0:
                pd_opt += 1
            opt_distances.append(dist_opt)
                
            # Without direction info
            opt_nodir_est = opt.estimate_source(use_directions=False)
            if opt_nodir_est == -1:
                print('NO BUENO')
                num_singular += 1
                continue
            dist_opt_nodir = networkx.shortest_path_length(opt.graph,source, opt_nodir_est)
            if dist_opt_nodir == 0:
                pd_opt_nodir += 1
            opt_nodir_distances.append(dist_opt_nodir)
            
            
            # # Entropy estimator
            # entropy = estimation.EntropyEstimator(adjacency, malicious_nodes, timestamps)
            # entropy_est = entropy.estimate_source()
            # # print('Sum distance estimate is: ', entropy_est)
            # # print('True source is :', source)
            # # print('Distance from the true source is :', distances[entropy_est])
            # entropy_distances.append(distances[entropy_est])
            
            # Random estimator
            rand_est = random.choice([i for i in range(len(adjacency)) if i not in malicious_nodes])
            rand_dist = networkx.shortest_path_length(opt.graph,source, rand_est)
            rand_distances.append(rand_dist)
            # print('Rand estimate is' ,rand_est, 'Random distance from the true source is :', rand_dist)
            if rand_dist == 0:
                pd_rand += 1

           # # Stupid Estimator
            # opt = estimation.StupidEstimator(adjacency, malicious_nodes, timestamps)
            # opt_est = opt.estimate_source()
            # if opt_est == -1:
                # print('NO BUENO')
                # num_singular += 1
                # continue
            # print('Stupid estimate is: ', opt_est)
            # print('True source is :', source)
            # print('Distance from the true source is :', networkx.shortest_path_length(opt.graph,source, opt_est))
            
            # Nearest Spy Estimator
            spy = estimation.FirstSpyEstimator(adjacency, malicious_nodes, timestamps, infectors)
            spy_est = spy.estimate_source()
            if spy_est == -1:
                print('NO BUENO')
                num_singular += 1
                continue
            spy_dist = networkx.shortest_path_length(opt.graph,source, spy_est)
            # print('Nearest-spy estimate is: ', spy_est,'True source is :', source,'Distance from the true source is :', spy_dist)
            spy_distances.append(spy_dist)
            if spy_dist == 0:
                pd_spy += 1
                
            # Without direction info
            spy_nodir_est = spy.estimate_source(use_directions=False)
            if spy_nodir_est == -1:
                print('NO BUENO')
                num_singular += 1
                continue
            spy_nodir_dist = networkx.shortest_path_length(opt.graph,source, spy_nodir_est)
            # print('Nearest-spy estimate is: ', spy_est,'True source is :', source,'Distance from the true source is :', spy_dist)
            spy_nodir_distances.append(spy_nodir_dist)
            if spy_nodir_dist == 0:
                pd_spy_nodir += 1
            
        # # print('The fraction of singular matrices is ', num_singular / float(trials))
        # # print('Distances are: ', opt_distances)
        # print('mean optimal distance is: ', sum(opt_distances)/float(len(opt_distances)))
        # # print('mean entropy distance is: ', sum(entropy_distances)/float(len(entropy_distances)))
        # print('mean random distance is: ', sum(rand_distances)/float(len(rand_distances)))
        # print('mean nearest-spy distance is: ', sum(spy_distances)/float(len(spy_distances)))
        # print('Optimal Pd = ',float(pd_opt) / trials)
        # print('Random Pd = ',float(pd_rand) / trials)
        # print('Spy Pd = ',float(pd_spy) / trials)
        
        pds_opt.append(float(pd_opt) / trials)
        print('the probabiliy of detection is ',float(pd_opt) / trials)
        pds_opt_nodir.append(float(pd_opt_nodir) / trials)
        print('the probabiliy of detection without direction info is ',float(pd_opt_nodir) / trials)
        pds_spy.append(float(pd_spy) / trials)
        print('the probabiliy of spy detection is ',float(pd_spy) / trials)
        pds_spy_nodir.append(float(pd_spy_nodir) / trials)
        print('the probabiliy of spy detection without direction info is ',float(pd_spy_nodir) / trials)
        pds_rand.append(float(pd_rand) / trials)
        
        hops_opt.append(sum(opt_distances)/float(len(opt_distances)))
        hops_opt_nodir.append(sum(opt_nodir_distances)/float(len(opt_nodir_distances)))
        hops_spy.append(sum(spy_distances)/float(len(spy_distances)))
        hops_spy_nodir.append(sum(spy_nodir_distances)/float(len(spy_nodir_distances)))
        hops_rand.append(sum(rand_distances)/float(len(rand_distances)))
        
    
        #write_filename = 'data/' + graph_size + '/results/res.mat'
        write_filename = 'results/res.mat'
        savemat(write_filename, dict(num_singular = num_singular,
                                     trials = trials, 
                                     pds_opt = pds_opt,
                                     pds_opt_nodir = pds_opt_nodir,
                                     pds_spy = pds_spy,
                                     pds_spy_nodir = pds_spy_nodir,
                                     pds_rand = pds_rand,
                                     hops_opt = hops_opt,
                                     hops_opt_nodir = hops_opt_nodir,
                                     hops_spy = hops_spy,
                                     hops_spy_nodir = hops_spy_nodir,
                                     hops_rand = hops_rand))

    
    # # Jordan estimator
    # jordan = estimation.JordanEstimator(adjacency, malicious_nodes, timestamps)
    # jordan_est = jordan.estimate_source()
    # print('Jordan estimate is: ', jordan_est)
    # print('True source is :', source)
    # print('Distance from the true source is :', distances[jordan_est])

    
    # # Entropy estimator
    # entropy = estimation.EntropyEstimator(adjacency, malicious_nodes, timestamps)
    # entropy_est = entropy.estimate_source()
    # print('Sum distance estimate is: ', entropy_est)
    # print('True source is :', source)
    # print('Distance from the true source is :', distances[entropy_est])
    
    
    # # Sum distance estimator
    # sum_dist = estimation.SumDistanceEstimator(adjacency, malicious_nodes, timestamps)
    # sum_dist_est = sum_dist.estimate_source(4000)
    # print('Sum distance estimate is: ', sum_dist_est)
    # print('True source is :', source)
    # print('Distance from the true source is :', distances[sum_dist_est])
    
