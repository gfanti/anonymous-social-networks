import Queue
import networkx as nx
import random
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

from node import *

MALICIOUS = "bad"
HONEST = "good"
ATTRNAME = "node_type"

ATTROBJ = "object"

# discrete time simulator
class Simulator(object):
    def __init__(self, end_time = 0):
        self.event_queue = Queue.PriorityQueue()
        self.current_time = 0
        self.end_time = end_time
        
    # event should be a callback function
    def schedule_event(self, time_delta, event):
        self.event_queue.put_nowait((self.current_time + time_delta, event))

    def run(self, end_time):
        self.end_time = end_time
        while True:
            if self.event_queue.empty():
                break
            if self.current_time >= self.end_time:
                break

            (t, event) = self.event_queue.get_nowait()
            #print t
            self.current_time = max(t, self.current_time + 1)
            event()

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


class Graph(object):
    def __init__(self, generator, sim):
        self.g = generator.generate()
        self.honest_nodes = {}
        self.malicious_nodes = {}
        self.sim = sim
        self.reset()

    def reset(self):
        for i in xrange(len(self.g.nodes())):
            self.g.node[i][ATTRNAME] = HONEST
            self.g.node[i][ATTROBJ] = None

    def infect_random(self, percent):
        all_nodes = self.g.nodes()
        random.shuffle(all_nodes)
        idx = math.floor((percent / 100.0) * len(all_nodes))

        for cur_id in all_nodes:
            if cur_id < idx:
                self.g.node[cur_id][ATTRNAME] = MALICIOUS
                self.g.node[cur_id][ATTROBJ] = MaliciousNode(cur_id, self.sim)
            else:
                self.g.node[cur_id][ATTRNAME] = HONEST
                self.g.node[cur_id][ATTROBJ] = HonestNode(cur_id, self.sim)

        # add neighbors
        for cur_id in all_nodes:
            node = self.g.node[cur_id][ATTROBJ]
            neighbors = self.g.neighbors(cur_id)
            for n in neighbors:
                node.add_neighbor(self.g.node[n][ATTROBJ])

    def nodes(self):
        return self.g.nodes(data=True)
        
    def get_random_node(self, t = None):
        all_nodes = self.g.nodes()
        nodes = []
        
        if t is None:
            nodes = all_nodes
        else:
            for n in all_nodes:
                if self.g.node[n][ATTRNAME] == t:
                    nodes.append(self.g.node[n])

        ret = random.sample(nodes, 1)[0]
        return ret[ATTROBJ]

    # for debugging/visualization purposes
    def draw_graph(self):
        G = self.g
        pos = nx.spring_layout(G)
        nl = [x[0] for x in G.nodes(data=True) if x[1][ATTRNAME] == HONEST]
        nx.draw_networkx_nodes(G,pos,nodelist=nl,node_color="#A0CBE2")
        nl = [x[0] for x in G.nodes(data=True) if x[1][ATTRNAME] == MALICIOUS]
        nx.draw_networkx_nodes(G,pos,nodelist=nl,node_color="red")
        nx.draw_networkx_edges(G,pos,width=1.0,alpha=0.5)
        labels = {}
        for x in G.nodes(data=True):
            labels[x[0]] = x[0]
        nx.draw_networkx_labels(G, pos, labels)
        plt.show()
        
class Simulation(object):
    def __init__(self, graph = None):
        self.sim = Simulator()
        self.graph = Graph(WaxmanGraphGenerator(100, 20), self.sim)
        self.graph.infect_random(10)

    def start(self):
        n = self.graph.get_random_node(HONEST)
        self.sim.schedule_event(0, n.generate_message)
        for n in self.graph.nodes():
            self.sim.schedule_event(0, n[1][ATTROBJ].loop)
        
        self.sim.run(10000)
        for n in self.graph.nodes():
            if n[1][ATTRNAME] == MALICIOUS:
                #print type(n[1][ATTROBJ])
                print n[1][ATTROBJ].intercepted_messages
        
Simulation().start()

