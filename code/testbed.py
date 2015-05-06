#testbed
from spreadparser import Parser
import estimation
import random
from scipy.io import savemat
import sys
import networkx

if __name__ == '__main__':


    opt_distances = []
    pd_opt = 0
    pd_rand = 0
    pd_spy = 0
    entropy_distances = []
    rand_distances = []
    spy_distances = []
    num_singular = 0
    trials = 200
    num_nodes = 100
    percent_malicious = 90
    graph_size = 'N' + str(num_nodes) + '_BA'
    directory = 'data/' + graph_size + '/malicious_' + str(percent_malicious) + '/'
    for i in range(trials):
        parser = Parser( directory + 'output' + str(i+1))
        # parser = Parser(sys.argv[1])
        source, adjacency, malicious_nodes, timestamps = parser.parse_file()
        
        e = estimation.Estimator(adjacency, malicious_nodes, timestamps)
        
        print('the diameter is ', e.get_diameter())
        # exit(0)
        
        # Optimal estimator
        print('malicious nodes are',malicious_nodes)
        opt = estimation.OptimalEstimator(adjacency, malicious_nodes, timestamps)
        opt_est = opt.estimate_source()
        if opt_est == -1:
            print('NO BUENO')
            num_singular += 1
            continue
        dist_opt = networkx.shortest_path_length(opt.graph,source, opt_est)
        if dist_opt == 0:
            pd_opt += 1
        print('Optimal estimate is: ', opt_est,'True source is :', source, 'Distance from the true source is :', dist_opt)
        
        opt_distances.append(dist_opt)
        
        
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
        print('Rand estimate is' ,rand_est, 'Random distance from the true source is :', rand_dist)
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
        spy = estimation.FirstSpyEstimator(adjacency, malicious_nodes, timestamps)
        spy_est = spy.estimate_source()
        if spy_est == -1:
            print('NO BUENO')
            num_singular += 1
            continue
        spy_dist = networkx.shortest_path_length(opt.graph,source, spy_est)
        print('Nearest-spy estimate is: ', spy_est,'True source is :', source,'Distance from the true source is :', spy_dist)
        spy_distances.append(spy_dist)
        if spy_dist == 0:
            pd_spy += 1
        
    # print('The fraction of singular matrices is ', num_singular / float(trials))
    # print('Distances are: ', opt_distances)
    print('mean optimal distance is: ', sum(opt_distances)/float(len(opt_distances)))
    # print('mean entropy distance is: ', sum(entropy_distances)/float(len(entropy_distances)))
    print('mean random distance is: ', sum(rand_distances)/float(len(rand_distances)))
    print('mean nearest-spy distance is: ', sum(spy_distances)/float(len(spy_distances)))
    print('Optimal Pd = ',float(pd_opt) / trials)
    print('Random Pd = ',float(pd_rand) / trials)
    print('Spy Pd = ',float(pd_spy) / trials)
    
    # write_filename = 'data/' + graph_size + '/results/malicious_' + str(percent_malicious) + '.mat'
    # savemat(write_filename, dict(num_singular = num_singular,
                                 # trials = trials, 
                                 # opt_distances = opt_distances,
                                 # entropy_distances = entropy_distances,
                                 # rand_distances = rand_distances))

    
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
    
