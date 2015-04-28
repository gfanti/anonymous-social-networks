import networkx as nx

class RandomGraphGenerator(object):
    def __init__(self, num_nodes, edges_conn):
        self.num_nodes = num_nodes
        self.edges_conn = edges_conn

    def generate(self):
        g = nx.erdos_renyi_graph(self.num_nodes, self.edges_conn)
        return g

class WaxmanGraphGenerator(object):
    def __init__(self, num_nodes, edges_conn):
        self.num_nodes = num_nodes
        self.edges_conn = edges_conn

    def generate(self):
        g = nx.waxman_graph(self.num_nodes, alpha=0.6)
        return g

class WattsStrogatzGraphGenerator(object):
    def __init__(self, n, k, p):
        self.n = n
        self.k = k
        self.p = p

    def generate(self):
        g = nx.connected_watts_strogatz_graph(self.n, self.k, self.p)
        return g

# Barabasi-Albert graph
class BAGraphGenerator(object):
    def __init__(self, n, m):
        self.n = n
        self.m = m

    def generate(self):
        g = nx.barabasi_albert_graph(self.n, self.m)
        return g

# Balanced r-tree 
class BTGraphGenerator(object):
    def __init__(self, r, h):
        self.r = r
        self.h = h

    def generate(self):
        g = nx.balanced_tree(self.r, self.h)
        return g        
        
# Power-law tree graph  (doesn't work well :( )
class PLTreeGraphGenerator(object):
    def __init__(self, n):
        self.n = n

    def generate(self):
        g = nx.random_powerlaw_tree(self.n)
        return g