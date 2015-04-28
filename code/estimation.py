# estimation.py
import random
import math
import numpy as np
from numpy.linalg import inv, pinv
import heapq

class Estimator(object):
    def __init__(self, adjacency, malicious_nodes, timestamps):
        self.adjacency = adjacency
        self.malicious_nodes = malicious_nodes
        self.timestamps = timestamps
        
    def estimate_source(self):
        pass
        
    def get_distances(self, source):
        visited, queue = set(), [source]
        distances = [0 for i in range(len(self.adjacency))]
        while queue and (0 in [distances[j] for j in self.malicious_nodes]):
            vertex = queue.pop(0)
            if vertex not in visited:
                visited.add(vertex)
                queue.extend(self.adjacency[vertex] - visited)
                vertex_dist = distances[vertex]
                for i in (self.adjacency[vertex] - visited):
                    distances[i] = vertex_dist + 1
        return distances
        
    def get_diameter(self):
        # computes the diameter of the adjacency matrix
        max_dist = 0
        for node in range(len(self.adjacency)):
            distances = self.get_distances(node)
            if max(distances) > max_dist:
                max_dist = max(distances)
        return max_dist
        
    def get_spanning_tree(self, node):
        num_nodes = len(self.adjacency)
        sp_adjacency = [set() for i in range(num_nodes)]
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
                    sp_adjacency[vertex].add(i)
                    sp_adjacency[i].add(vertex)
        return sp_adjacency
        
    def dijkstra(self, source, destination, adjacency = None):
        ''' Return predecessors and min distance if there exists a shortest path 
            from s to t; Otherwise, return None '''
        Q = []     # priority queue of items; item is mutable.
        d = {source: 0} # vertex -> minimal distance
        Qd = {}    # vertex -> [d[v], parent_v, v]
        p = {}     # predecessor
        visited_set = set([source])

        if not adjacency:
            adjacency = self.adjacency
        
        for v in adjacency[source]:
            d[v] = 1
            item = [d[v], source, v]
            heapq.heappush(Q, item)
            Qd[v] = item

        while Q:
            cost, parent, u = heapq.heappop(Q)
            if u not in visited_set:
                p[u]= parent
                visited_set.add(u)
                if u == destination:
                    # p is the predecessors
                    c = destination
                    path = [c]
                    while p.get(c):
                        path.insert(0, p[c])
                        c = p[c]
                    return path
                for v in adjacency[u]:
                    if d.get(v):
                        if d[v] > 1 + d[u]:
                            d[v] =  1 + d[u]
                            Qd[v][0] = d[v]    # decrease key
                            Qd[v][1] = u       # update predecessor
                            heapq._siftdown(Q, 0, Q.index(Qd[v]))
                    else:
                        d[v] = 1 + d[u]
                        item = [d[v], u, v]
                        heapq.heappush(Q, item)
                        Qd[v] = item
        return None
        
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

class OptimalEstimator(Estimator):
    def estimate_source(self):
        # Sums the distance to the unvisited nodes and visited nodes at time_t
        max_likelihood = None
        max_index = -1
        d = np.diff(self.timestamps)
        num_spies = len(self.malicious_nodes)
        # First compute the paths between spy 1 and the rest
                            
        for node in range(len(self.adjacency)):
            if node in self.malicious_nodes:
                continue
            sum_distance = 0
            distances = self.get_distances(node)
            # 2 is the mean delay if a message gets forwarded
            mu = np.array([2*(distances[self.malicious_nodes[k+1]] - distances[self.malicious_nodes[0]]) for k in range(num_spies-1)])
            mu.shape = (1,len(mu))
            Lambda_inv = self.compute_lambda_inv(node)
            # subtract distance from nodes that have seen the message already
            d_norm = np.array([item_d - 0.5*item_mu for (item_d, item_mu) in zip(d, mu)])
            d_norm = np.transpose(d_norm)
            likelihood = float(np.dot(np.dot(mu, Lambda_inv), d_norm))
            if (max_likelihood is None) or (max_likelihood < likelihood):
                max_likelihood = likelihood
                max_index = node
        return max_index
        
    def compute_lambda_inv(self, node):
        num_spies = len(self.malicious_nodes)
        Lambda = np.matrix(np.zeros((num_spies-1, num_spies-1)))
        spy_distances = self.get_distances(self.malicious_nodes[0])
                
        paths = []
        spanning_tree = self.get_spanning_tree(node)
        for i in range(num_spies-1):
            source = self.malicious_nodes[0]
            destination = self.malicious_nodes[i+1]
            path = self.dijkstra(source, destination, spanning_tree)
            paths.append(set(path))
        for i in range(num_spies-1):
            for j in range(num_spies-1):
                if i == j:
                    Lambda[i,j] = spy_distances[self.malicious_nodes[i+1]]
                else:
                    Lambda[i,j] = len(paths[i].intersection(paths[j]))
                    Lambda[j,i] = Lambda[i,j]
        try:
            Lambda_inv = inv(Lambda)
        except:
            # print('matrix was not invertible.')
            # return max_index
            Lambda_inv = pinv(Lambda)
        return Lambda_inv
        
    def get_shortest_path(self, start, end, path=[]):
        # Returns a path of nodes from source to destination (inclusive)
        path = path + [start]
        if start == end:
            return path
        shortest = None
        for node in self.adjacency[start]:
            if node not in path:
                newpath = self.get_shortest_path(node, end, path)
                if newpath:
                    if not shortest or len(newpath) < len(shortest):
                        shortest = newpath
                        print('shortest is', shortest)
        return shortest
        
        visited, queue = set(), [(source, set([source]))]
        distances = [0 for i in range(len(self.adjacency))]
        while queue and (0 in [distances[j] for j in self.malicious_nodes]):
            vertex, path = queue.pop(0)
            if vertex not in visited:
                visited.add(vertex)
                queue.extend(self.adjacency[vertex] - visited)
                vertex_dist = distances[vertex]
                for i in (self.adjacency[vertex] - visited):
                    distances[i] = vertex_dist + 1
        return distances
        
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
        
    