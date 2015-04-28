import Queue
import networkx as nx
import random
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

from node import *
from generators import *

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

class Graph(object):
    def __init__(self, generator, sim):
        self.g = generator.generate()
        self.honest_nodes = {}
        self.malicious_nodes = {}
        self.sim = sim
        self.reset()

    def reset(self):
        for i in self.g.nodes():
            self.g.node[i][ATTRNAME] = HONEST
            self.g.node[i][ATTROBJ] = None

    def infect_random(self, percent):
        all_nodes = self.g.nodes()
        random.shuffle(all_nodes)
        idx = math.floor((percent / 100.0) * len(all_nodes))

        for i in range(len(all_nodes)):
            cur_id = all_nodes[i]
            if i < idx:
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

    def print_graph(self):
        for n in self.g.nodes(data=True):
            print n[0], n[1][ATTRNAME]
            print self.g.neighbors(n[0])
        
class Simulation(object):
    def __init__(self, generator, graph = None):
        self.sim = Simulator()
        self.graph = Graph(generator, self.sim)
        self.graph.infect_random(60)

    def start(self):
        self.graph.print_graph()
        n = self.graph.get_random_node(HONEST)
        self.sim.schedule_event(0, n.generate_message)
        for n in self.graph.nodes():
            self.sim.schedule_event(0, n[1][ATTROBJ].loop)
        
        self.sim.run(10000)
        print "Simulation done"
        for n in self.graph.nodes():
            if n[1][ATTRNAME] == MALICIOUS:
                #print type(n[1][ATTROBJ])
                o = n[1][ATTROBJ]
                print o.node_id, o.intercepted_messages
              
ggen = BTGraphGenerator(2, 4)
#ggen = FacebookDataGenerator(300)
Simulation(ggen).start()
