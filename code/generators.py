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

class FacebookDataGenerator(object):
    def __init__(self, n):
        self.n = n

    def generate(self):
        # generate an n-node graph from the facebook data
        g = nx.Graph()
        f = open("./data/facebook-links.txt")
        sampled_nodes = set([])
        for line in f:
            l = line.split('\n')[0].split('\t')
            node_0 = int(l[0])
            node_1 = int(l[1])
            if node_0 < self.n and node_1 < self.n:
                g.add_edge(node_0, node_1)
                sampled_nodes.add(node_0)
            if len(sampled_nodes) == self.n:
                break
        return g
            
