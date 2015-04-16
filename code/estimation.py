# estimation.py
import random

class Estimator(object):
    def __init__(self, adjacency, malicious_nodes, timestamps):
        self.adjacency = adjacency
        self.malicious_nodes = malicious_nodes
        self.timestamps = timestamps
        
    def estimate_source(self):
        pass
        
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
        return jordan_center, jordan_dist
    
    def compute_jordan_centrality(self, source):
        # Computes the jordan centrality of a single node 'source', as viewed by the malicious nodes
        
        # compute the distances to all malicious nodes
        distances = self.get_distances(source)
        
        # Extract the longest distance to any one of the malicious nodes
        jordan_dist = max([distances[j] for j in self.malicious_nodes])
        print("The distance of ", source, " is ", jordan_dist, "\n")
        return jordan_dist
        
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