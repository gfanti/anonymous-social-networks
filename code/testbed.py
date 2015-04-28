#testbed
from spreadparser import Parser
import estimation
import random
from scipy.io import savemat

if __name__ == '__main__':


    opt_distances = []
    entropy_distances = []
    rand_distances = []
    num_singular = 0
    trials = 20
    num_nodes = 100
    percent_malicious = 60
    graph_size = 'N' + str(num_nodes) + '_BA'
    directory = 'data/' + graph_size + '/malicious_' + str(percent_malicious) + '/'
    for i in range(trials):
        parser = Parser( directory + 'output' + str(i+1))
        source, adjacency, malicious_nodes, timestamps = parser.parse_file()
        
        e = estimation.Estimator(adjacency, malicious_nodes, timestamps)
        distances = e.get_distances(source)
        
        print('the diameter is ', e.get_diameter())
        # exit(0)
        
        # Optimal estimator
        opt = estimation.OptimalEstimator(adjacency, malicious_nodes, timestamps)
        opt_est = opt.estimate_source()
        if opt_est == -1:
            print('NO BUENO')
            num_singular += 1
            continue
        print('Optimal estimate is: ', opt_est)
        print('True source is :', source)
        print('Distance from the true source is :', distances[opt_est])
        
        opt_distances.append(distances[opt_est])
        
        
        # Entropy estimator
        entropy = estimation.EntropyEstimator(adjacency, malicious_nodes, timestamps)
        entropy_est = entropy.estimate_source()
        # print('Sum distance estimate is: ', entropy_est)
        # print('True source is :', source)
        # print('Distance from the true source is :', distances[entropy_est])
        entropy_distances.append(distances[entropy_est])
        
        # Random estimator
        rand_est = random.randint(0,len(adjacency)-1)
        rand_distances.append(distances[rand_est])
        print('Random distance from the true source is :', distances[rand_est])
    
    print('The fraction of singular matrices is ', num_singular / float(trials))
    print('Distances are: ', opt_distances)
    print('mean optimal distance is: ', sum(opt_distances)/float(len(opt_distances)))
    print('mean entropy distance is: ', sum(entropy_distances)/float(len(entropy_distances)))
    print('mean random distance is: ', sum(rand_distances)/float(len(rand_distances)))
    
    write_filename = 'data/' + graph_size + '/results/malicious_' + str(percent_malicious) + '.mat'
    savemat(write_filename, dict(num_singular = num_singular,
                                 trials = trials, 
                                 opt_distances = opt_distances,
                                 entropy_distances = entropy_distances,
                                 rand_distances = rand_distances))

    
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
    