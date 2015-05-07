# estimation.py
import random
import math
import numpy as np
from numpy.linalg import inv, pinv
import heapq
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt

class Estimator(object):
    def __init__(self, adjacency, malicious_nodes, timestamps, active_nodes = None):
        self.adjacency = adjacency
        self.malicious_nodes = malicious_nodes
        self.timestamps = timestamps
        self.graph = nx.Graph()
        
        # Populate the active nodes
        if active_nodes is None:
            self.active_nodes = [1 for i in range(len(adjacency))]
        else:
            self.active_nodes = active_nodes
        # Populate the graph
        for idx in range(len(self.adjacency)):
            edges = self.adjacency[idx]
            for e in edges:
                self.graph.add_edge(idx, e)
        
    def estimate_source(self):
        pass
        
    def get_diameter(self):
        ''' Returns the diameter of the graph'''
        # computes the diameter of the adjacency matrix
        return nx.diameter(self.graph)
    
    def draw_graph(self):
        G = self.graph
        pos = nx.spring_layout(G)
        nl = [x for x in G.nodes() if x not in self.malicious_nodes]
        nx.draw_networkx_nodes(G,pos,nodelist=nl,node_color="#A0CBE2")
        nl = [x for x in G.nodes() if x in self.malicious_nodes]
        nx.draw_networkx_nodes(G,pos,nodelist=nl,node_color="red")
        nx.draw_networkx_edges(G,pos,width=1.0,alpha=0.5)
        labels = {}
        for x in G.nodes(data=True):
            labels[x[0]] = x[0]
        nx.draw_networkx_labels(G, pos, labels)
        plt.show()
        
    def get_spanning_tree(self, node):
        ''' Returns a networkx spanning tree of the adjacency matrix
        rooted at node'''
        num_nodes = len(self.adjacency)
        # sp_adjacency = [set() for i in range(num_nodes)]
        G = nx.Graph()
        for vertex in range(len(self.adjacency)):
            G.add_node(vertex)
        nodes = set([i for i in range(num_nodes)])
        visited, queue = set(), [node]
        while queue and nodes:
            vertex = queue.pop(0)
            if vertex not in visited:
                nodes.remove(vertex)
                visited.add(vertex)
                queue.extend(self.adjacency[vertex] - visited)
                # Fix the adjacency matrix here!
                for i in (self.adjacency[vertex] - visited):
                    # sp_adjacency[vertex].add(i)
                    # sp_adjacency[i].add(vertex)
                    G.add_edge(vertex, i)
        # return sp_adjacency

        G = nx.bfs_tree(self.graph, node).to_undirected()
        return G
                
class OptimalEstimator(Estimator):
    def __init__(self, adjacency, malicious_nodes, timestamps, active_nodes = None):
        super(OptimalEstimator, self).__init__(adjacency, malicious_nodes, timestamps, active_nodes)

        min_ = None
        for i in range(len(self.malicious_nodes)):
            r = self.graph.neighbors(self.malicious_nodes[i])
            r = len(r)
            #r = timestamps[i]
            if min_ is None:
                min_ = r
                self.ref = i
            elif min_ > r:
                min_ = r
                self.ref = i

        temp = timestamps[0]
        timestamps[0] = timestamps[self.ref]
        timestamps[self.ref] = temp

        temp = self.malicious_nodes[0]
        self.malicious_nodes[0] = self.malicious_nodes[self.ref]
        self.malicious_nodes[self.ref] = temp
        #self.ref = 0

    def estimate_source(self):
        # Sums the distance to the unvisited nodes and visited nodes at time_t
        max_likelihood = None
        max_indices = []
        num_spies = len(self.malicious_nodes)
        # d = np.diff(self.timestamps)
        d = np.array([self.timestamps[k+1] - self.timestamps[0] for k in range(num_spies - 1)])
        # First compute the paths between spy 1 and the rest
        # for i in range(num_spies):
            # print('Spy ',self.malicious_nodes[i],': ', self.timestamps[i])
        # print('timestamps are ', self.timestamps)
        # count = 0
        for node in range(len(self.adjacency)):
            if (node in self.malicious_nodes) or (self.active_nodes[node] == -1):
                continue
            
            # spanning_tree = self.get_spanning_tree(node)
            # inconsistent = False
            # for sp_node in self.malicious_nodes:
                # max_timestamp = 0.0
                # p = nx.shortest_path(spanning_tree,node, sp_node)
                # print('p',p,'node',node,'sp_node',sp_node)
                # for i in [y for y in p if y in self.malicious_nodes]:
                    # if self.timestamps[i] < max_timestamp:
                        # inconsistent = True
                        # break
                    # else:
                        # max_timestamp = self.timestamps[i]
                # if inconsistent:
                    # break
            # if inconsistent:
                # break
            # count += 1
                    
            # distances = self.get_distances(node)
            # 2 is the mean delay if a message gets forwarded
            # mu = np.array([2*(distances[self.malicious_nodes[k+1]] - distances[self.malicious_nodes[0]]) for k in range(num_spies-1)])
            mu = np.array([2.0*(nx.shortest_path_length(self.graph, node, self.malicious_nodes[k+1]) - 
                                nx.shortest_path_length(self.graph, node, self.malicious_nodes[0])) for k in range(num_spies-1)])
            mu.shape = (1,len(mu))
            # print('mu is ', mu, 'd is ',d)
            Lambda_inv, Lambda = self.compute_lambda_inv(node)
            # subtract distance from nodes that have seen the message already
            # d_norm = np.array([item_d - 0.5*item_mu for (item_d, item_mu) in zip(d, mu)])
            # d_norm = np.array([item_d - item_mu for (item_d, item_mu) in zip(d, mu)])
            # d_norm = np.transpose(d_norm)
            d_norm = []
            for idx in range(len(d)):
                d_norm.append(d[idx] - 0.5 * mu[0,idx])
            d_norm = np.transpose(np.array(d_norm))
            # likelihood = float(np.dot(np.dot(mu, Lambda_inv), d_norm))
            likelihood = np.exp(-0.5 * np.dot(np.dot(Lambda_inv, d_norm), d_norm)) / pow(np.linalg.det(Lambda), 0.5)
            
            # print('Node ', node,': likelihood is ', likelihood)
            if (max_likelihood is None) or (max_likelihood < likelihood):
                max_likelihood = likelihood
                max_indices = [node]
            elif (max_likelihood == likelihood):
                max_indices.append(node)
        # print('the candidates are ', max_indices)
        # print('the spies are ', self.malicious_nodes)
        return random.choice(max_indices)
        
    def compute_lambda_inv(self, node):
        num_spies = len(self.malicious_nodes)
        Lambda = np.matrix(np.zeros((num_spies-1, num_spies-1)))
        spanning_tree = self.get_spanning_tree(node)
        # distances = self.get_distances(self.malicious_nodes[0])
        # spy_distances = [distances[i] for i in self.malicious_nodes]
                
        paths = []
        for i in range(num_spies-1):
            source = self.malicious_nodes[0]
            destination = self.malicious_nodes[i+1]
            # path = self.dijkstra(source, destination, spanning_tree)
            path = nx.shortest_path(spanning_tree, source, destination)
            #path.pop(0)
            # print('path is ', path)
            # print('original adjacency is ', self.adjacency)
            # print('dijstra result from', source, ' to ', destination, ' gives ', path)
            #paths.append(set(path))
            paths.append(path)
        for i in range(num_spies-1):
            for j in range(num_spies-1):
                if i == j:
                    # Lambda[i,j] = spy_distances[i+1]
                    Lambda[i,j] = nx.shortest_path_length(spanning_tree,self.malicious_nodes[0],self.malicious_nodes[i+1])
                else:
                    p_i = zip(paths[i], paths[i][1:])
                    p_j = zip(paths[j], paths[j][1:])

                    count = 0
                    for e1 in p_i:
                        for e2 in p_j:
                            if (e1[0] == e2[0] and e1[1] == e2[1]) or (e1[1] == e2[0] and e1[0] == e2[1]):
                                count += 1

                    Lambda[i,j] = count
                    #Lambda[i,j] = len(paths[i].intersection(paths[j]))
                    Lambda[j,i] = Lambda[i,j]
        # print('Adjacency: ', self.adjacency)
        # print('Spies: ', self.malicious_nodes)
        # print('Spy_times: ', self.timestamps)
        # print('Lambda: ', Lambda)
        Lambda = 0.5**2 * Lambda
        try:
            Lambda_inv = inv(Lambda)
        except:
            # print('matrix was not invertible.')
            # return max_index
            Lambda_inv = pinv(Lambda)
        return Lambda_inv, Lambda

class FirstSpyEstimator(Estimator):
    def estimate_source(self):
        # Picks a random neighbor of the first spy to receive the message
        # print(self.timestamps)
        # print('timestamps 0',self.timestamps[0],self.malicious_nodes[0],'adj:',self.adjacency[self.malicious_nodes[0]])
        estimate = random.randint(0, len(self.adjacency)-1)
        for spy in self.malicious_nodes:
            options = [option for option in self.adjacency[spy] if option not in self.malicious_nodes]
            if options:
                estimate = random.choice(options)
                break
        return estimate
        
class SumDistanceEstimator(Estimator):
    def estimate_source(self, time_t):
        # Sums the distance to the unvisited nodes and visited nodes at time_t
        max_sum_distance = -100000
        max_index = -1
        for node in range(len(self.adjacency)):
            sum_distance = 0
            distances = self.get_distances(node)
            # subtract distance from nodes that have seen the message already
            sum_distance -= sum([distances[i] for i in range(len(self.malicious_nodes)) if self.timestamps[i] <= time_t])
            # add the distance to nodes that did not see the message yet
            sum_distance += sum([distances[i] for i in range(len(self.malicious_nodes)) if self.timestamps[i] > time_t])
            if max_sum_distance < sum_distance:
                max_index = node
                max_sum_distance = sum_distance
        return max_index
       
class EntropyEstimator(Estimator):
    def estimate_source(self):
        # Sums the distance to the unvisited nodes and visited nodes at time_t
        max_entropy = -1
        min_index = -1
        for node in range(len(self.adjacency)):
            distances = self.get_distances(node)
            # subtract distance from nodes that have seen the message already
            distribution = [distances[i]/self.timestamps[i] for i in range(len(self.malicious_nodes))]
            distribution = [item / sum(distribution) for item in distribution if item > 0]
            entropy = self.compute_entropy(distribution)
            if max_entropy < entropy:
                max_index = node
                max_entropy = entropy
        return max_index
        
    def compute_entropy(self, dist):
        # computes the entropy of distribution dist
        return sum([-p*math.log(p) for p in dist])
        
class JordanEstimator(Estimator):
    
    def estimate_source(self):
        # Computes the jordan estimate of the source
        jordan_dist = -1
        jordan_center = -1
        for node in range(len(self.adjacency)):
            dist = self.compute_jordan_centrality(node)
            if (jordan_dist == -1) or (dist < jordan_dist):
                jordan_dist = dist
                jordan_center = node
        print("The best estimate is ", jordan_center, " with a centrality of ", jordan_dist)
        return jordan_center
    
    def compute_jordan_centrality(self, source):
        # Computes the jordan centrality of a single node 'source', as viewed by the malicious nodes
        
        # compute the distances to all malicious nodes
        distances = self.get_distances(source)
        
        # Extract the longest distance to any one of the malicious nodes
        jordan_dist = max([distances[j] for j in self.malicious_nodes])
        return jordan_dist
        

class StupidEstimator(Estimator):
    def draw_graph(self, G, origin):
        plt.clf()
        pos = nx.spring_layout(G)

        nl = [x for x in G.nodes() if x not in self.malicious_nodes]
        nx.draw_networkx_nodes(G,pos,nodelist=nl,node_color="#A0CBE2")
        nl = [x for x in G.nodes() if x in self.malicious_nodes]
        nx.draw_networkx_nodes(G,pos,nodelist=nl,node_color="red")
        nl = [x for x in G.nodes() if x == origin]
        nx.draw_networkx_nodes(G,pos,nodelist=nl,node_color="black")
        nl = [x for x in G.nodes() if x == self.malicious_nodes[ref]]
        nx.draw_networkx_nodes(G,pos,nodelist=nl,node_color="green")

        nx.draw_networkx_edges(G,pos,width=1.0,alpha=0.5)
        labels = {}
        for x in G.nodes():
            if x in self.malicious_nodes:
                labels[x] = str(self.timestamps[self.malicious_nodes.index(x)])
        nx.draw_networkx_labels(G, pos, labels, font_size=16)
        plt.savefig('fig.png')
        raw_input()

    def estimate_source(self):
        # Sums the distance to the unvisited nodes and visited nodes at time_t
        max_likelihood = None
        max_indices = []
        num_spies = len(self.malicious_nodes)
        # d = np.diff(self.timestamps)
        #d = np.array([self.timestamps[k+1] - self.timestamps[ref] for k in range(num_spies - 1)])
        d = np.array([self.timestamps[k+1] for k in range(num_spies - 1)])
        # First compute the paths between spy 1 and the rest

        G = self.graph

        for node in range(len(self.adjacency)):
            if node in self.malicious_nodes:
                continue

            likelihood = 0
            min_time = None
            for m in self.malicious_nodes:
                t = self.timestamps[self.malicious_nodes.index(m)]
                if min_time is None:
                    min_time = t
                elif min_time > t:
                    min_time = t

            for m in self.malicious_nodes:
                distance = (len(nx.shortest_path(G, node, m)) - 1)
                actual_distance = self.timestamps[self.malicious_nodes.index(m)] - min_time
                likelihood += abs(actual_distance - distance)

            if (max_likelihood is None) or (max_likelihood > likelihood):
                max_likelihood = likelihood
                max_indices = [node]
            elif (max_likelihood == likelihood):
                max_indices.append(node)
        # print 'the candidates are ', max_indices
        # print 'the spies are ', self.malicious_nodes
        return random.choice(max_indices)

class MultiMessageOptimalEstimator(Estimator):
    def __init__(self, adjacency, malicious_nodes, all_timestamps, active_nodes = None):
        self.adjacency = adjacency
        self.malicious_nodes = malicious_nodes
        self.graph = nx.Graph()
        
        # Populate the active nodes
        if active_nodes is None:
            self.active_nodes = [1 for i in range(len(adjacency))]
        else:
            self.active_nodes = active_nodes
        # Populate the graph
        for idx in range(len(self.adjacency)):
            edges = self.adjacency[idx]
            for e in edges:
                self.graph.add_edge(idx, e)

        min_ = None
        for i in range(len(self.malicious_nodes)):
            r = self.graph.neighbors(self.malicious_nodes[i])
            r = len(r)
            #r = timestamps[i]
            if min_ is None:
                min_ = r
                self.ref = i
            elif min_ > r:
                min_ = r
                self.ref = i

        # for timestamps in all_timestamps:
        #     temp = timestamps[0]
        #     timestamps[0] = timestamps[self.ref]
        #     timestamps[self.ref] = temp

        # temp = self.malicious_nodes[0]
        # self.malicious_nodes[0] = self.malicious_nodes[self.ref]
        # self.malicious_nodes[self.ref] = temp

        self.ref = 0

        self.all_timestamps = all_timestamps
        self.likelihoods = {}

    def estimate_source(self):
        self.likelihoods = {}
        for timestamps in self.all_timestamps:
            self.estimate_source_(timestamps)

        max_likelihood = None
        max_node = None
        for n, l in self.likelihoods.iteritems():
            #print n, l, sum(l)
            if max_likelihood is None:
                max_likelihood = sum(l)
                max_node = n
            elif max_likelihood < sum(l):
                max_likelihood = sum(l)
                max_node = n

        return max_node
        
    def estimate_source_(self, timestamps):
        # Sums the distance to the unvisited nodes and visited nodes at time_t
        max_likelihood = None
        max_indices = []
        num_spies = len(self.malicious_nodes)
        # d = np.diff(self.timestamps)
        d = np.array([timestamps[k+1] - timestamps[0] for k in range(num_spies - 1)])
        # First compute the paths between spy 1 and the rest
        # for i in range(num_spies):
            # print('Spy ',self.malicious_nodes[i],': ', self.timestamps[i])
        # print('timestamps are ', self.timestamps)
        # count = 0
        for node in range(len(self.adjacency)):
            if (node in self.malicious_nodes) or (self.active_nodes[node] == -1):
                continue
            
            # spanning_tree = self.get_spanning_tree(node)
            # inconsistent = False
            # for sp_node in self.malicious_nodes:
                # max_timestamp = 0.0
                # p = nx.shortest_path(spanning_tree,node, sp_node)
                # print('p',p,'node',node,'sp_node',sp_node)
                # for i in [y for y in p if y in self.malicious_nodes]:
                    # if self.timestamps[i] < max_timestamp:
                        # inconsistent = True
                        # break
                    # else:
                        # max_timestamp = self.timestamps[i]
                # if inconsistent:
                    # break
            # if inconsistent:
                # break
            # count += 1
                    
            # distances = self.get_distances(node)
            # 2 is the mean delay if a message gets forwarded
            # mu = np.array([2*(distances[self.malicious_nodes[k+1]] - distances[self.malicious_nodes[0]]) for k in range(num_spies-1)])
            mu = np.array([2.0*(nx.shortest_path_length(self.graph, node, self.malicious_nodes[k+1]) - 
                                nx.shortest_path_length(self.graph, node, self.malicious_nodes[0])) for k in range(num_spies-1)])
            mu.shape = (1,len(mu))
            # print('mu is ', mu, 'd is ',d)
            Lambda, Lambda_inv = self.compute_lambda_inv(node)
            # subtract distance from nodes that have seen the message already
            # d_norm = np.array([item_d - 0.5*item_mu for (item_d, item_mu) in zip(d, mu)])
            # d_norm = np.array([item_d - item_mu for (item_d, item_mu) in zip(d, mu)])
            # d_norm = np.transpose(d_norm)
            d_norm = []
            for idx in range(len(d)):
                d_norm.append(d[idx] - 0.5 * mu[0,idx])
            d_norm = np.transpose(np.array(d_norm))
            # likelihood = float(np.dot(np.dot(mu, Lambda_inv), d_norm))
            likelihood = math.exp(-0.5 * float(np.dot(np.dot(np.transpose(d_norm), Lambda_inv), d_norm))) / (np.linalg.det(Lambda) ** 0.5)
            #likelihood = -0.5 * float(np.dot(np.dot(np.transpose(d_norm), Lambda_inv), d_norm))
            if node not in self.likelihoods:
                self.likelihoods[node] = []
            self.likelihoods[node].append(likelihood)

            # print('Node ', node,': likelihood is ', likelihood)
            if (max_likelihood is None) or (max_likelihood < likelihood):
                max_likelihood = likelihood
                max_indices = [node]
            elif (max_likelihood == likelihood):
                max_indices.append(node)
        # print('the candidates are ', max_indices)
        # print('the spies are ', self.malicious_nodes)
        return random.choice(max_indices)


        # # Sums the distance to the unvisited nodes and visited nodes at time_t
        # max_likelihood = None
        # max_indices = []
        # num_spies = len(self.malicious_nodes)
        # # d = np.diff(self.timestamps)
        # d = np.array([timestamps[k+1] - timestamps[0] for k in range(num_spies - 1)])
        # # First compute the paths between spy 1 and the rest
                            
        # for node in range(len(self.adjacency)):
        #     if (node in self.malicious_nodes) or (self.active_nodes[node] == -1):
        #         continue
        #     sum_distance = 0
        #     # distances = self.get_distances(node)
        #     # 2 is the mean delay if a message gets forwarded
        #     # mu = np.array([2*(distances[self.malicious_nodes[k+1]] - distances[self.malicious_nodes[0]]) for k in range(num_spies-1)])
        #     mu = np.array([2.0*(networkx.shortest_path_length(self.graph,node, self.malicious_nodes[k+1]) - 
        #                         networkx.shortest_path_length(self.graph,node, self.malicious_nodes[0])) for k in range(num_spies-1)])
        #     mu.shape = (1,len(mu))
        #     # print('timestamps are ', self.timestamps)
        #     # print('mu is ', mu, 'd is ',d)
        #     Lambda_inv = self.compute_lambda_inv(node)
        #     # subtract distance from nodes that have seen the message already
        #     d_norm = np.array([item_d - 0.5*item_mu for (item_d, item_mu) in zip(d, mu)])
        #     d_norm = np.transpose(d_norm)
        #     likelihood = float(np.dot(np.dot(mu, Lambda_inv), d_norm))
        #     # print('Node ', node,': likelihood is ', likelihood)
        #     if node not in self.likelihoods:
        #         self.likelihoods[node] = []
        #     self.likelihoods[node].append(likelihood)
        #     if (max_likelihood is None) or (max_likelihood < likelihood):
        #         max_likelihood = likelihood
        #         max_indices = [node]
        #     elif (max_likelihood == likelihood):
        #         max_indices.append(node)
        # # print('the candidates are ', max_indices)
        # # print('the spies are ', self.malicious_nodes)
        # return random.choice(max_indices)
        
    # def compute_lambda_inv(self, node):
    #     num_spies = len(self.malicious_nodes)
    #     Lambda = np.matrix(np.zeros((num_spies-1, num_spies-1)))
    #     spanning_tree = self.get_spanning_tree(node)
    #     # distances = self.get_distances(self.malicious_nodes[0])
    #     # spy_distances = [distances[i] for i in self.malicious_nodes]
                
    #     paths = []
    #     for i in range(num_spies-1):
    #         source = self.malicious_nodes[0]
    #         destination = self.malicious_nodes[i+1]
    #         # path = self.dijkstra(source, destination, spanning_tree)
    #         path = networkx.shortest_path(spanning_tree, source, destination)
    #         path.pop(0)
    #         # print('path is ', path)
    #         # print('original adjacency is ', self.adjacency)
    #         # print('dijstra result from', source, ' to ', destination, ' gives ', path)
    #         paths.append(set(path))
    #     for i in range(num_spies-1):
    #         for j in range(num_spies-1):
    #             if i == j:
    #                 # Lambda[i,j] = spy_distances[i+1]
    #                 Lambda[i,j] = networkx.shortest_path_length(spanning_tree,self.malicious_nodes[0],self.malicious_nodes[i+1])
    #             else:
    #                 Lambda[i,j] = len(paths[i].intersection(paths[j]))
    #                 Lambda[j,i] = Lambda[i,j]
    #     # print('Adjacency: ', self.adjacency)
    #     # print('Spies: ', self.malicious_nodes)
    #     # print('Spy_times: ', self.timestamps)
    #     # print('Lambda: ', Lambda)
    #     try:
    #         Lambda_inv = inv(Lambda)
    #     except:
    #         # print('matrix was not invertible.')
    #         # return max_index
    #         Lambda_inv = pinv(Lambda)
    #     return Lambda_inv

    def compute_lambda_inv(self, node):
        num_spies = len(self.malicious_nodes)
        Lambda = np.matrix(np.zeros((num_spies-1, num_spies-1)))
        spanning_tree = self.get_spanning_tree(node)
        # distances = self.get_distances(self.malicious_nodes[0])
        # spy_distances = [distances[i] for i in self.malicious_nodes]
                
        paths = []
        for i in range(num_spies-1):
            source = self.malicious_nodes[0]
            destination = self.malicious_nodes[i+1]
            # path = self.dijkstra(source, destination, spanning_tree)
            path = nx.shortest_path(spanning_tree, source, destination)
            #path.pop(0)
            # print('path is ', path)
            # print('original adjacency is ', self.adjacency)
            # print('dijstra result from', source, ' to ', destination, ' gives ', path)
            #paths.append(set(path))
            paths.append(path)
        for i in range(num_spies-1):
            for j in range(num_spies-1):
                if i == j:
                    # Lambda[i,j] = spy_distances[i+1]
                    Lambda[i,j] = nx.shortest_path_length(spanning_tree,self.malicious_nodes[0],self.malicious_nodes[i+1])
                else:
                    p_i = zip(paths[i], paths[i][1:])
                    p_j = zip(paths[j], paths[j][1:])

                    count = 0
                    for e1 in p_i:
                        for e2 in p_j:
                            if (e1[0] == e2[0] and e1[1] == e2[1]) or (e1[1] == e2[0] and e1[0] == e2[1]):
                                count += 1

                    Lambda[i,j] = count
                    #Lambda[i,j] = len(paths[i].intersection(paths[j]))
                    Lambda[j,i] = Lambda[i,j]
        # print('Adjacency: ', self.adjacency)
        # print('Spies: ', self.malicious_nodes)
        # print('Spy_times: ', self.timestamps)
        # print('Lambda: ', Lambda)
        Lambda = 0.5**2 * Lambda
        try:
            Lambda_inv = inv(Lambda)
        except:
            # print('matrix was not invertible.')
            # return max_index
            Lambda_inv = pinv(Lambda)
        return Lambda, Lambda_inv
