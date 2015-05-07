import Queue
import networkx as nx
import random
import math
import numpy as np

class NetworkLatency(object):
    @classmethod
    def next(cls, p):
        return np.random.normal(2, 0.5)
        #return np.random.geometric(p, 1)[0]

class Message(object):
    def __init__(self, true_source, source, content):
        self.id = random.getrandbits(64)
        self.content = content
        self.true_source = true_source
        self.source = source

# base node class
class Node(object):
    def __init__(self, node_id, sim):
        self.node_id = node_id
        self.sim = sim
        self.neighbors = []

        self.message_queue = Queue.Queue()
        self.past_messages = set([])

    def add_neighbor(self, n):
        self.neighbors.append(n)

    def send(self, m):
        if m.id in self.past_messages:
            return
        for n in self.neighbors:
            latency = NetworkLatency.next(0.5)
            #print "send:", self.sim.current_time, latency
            # schedule 
            #(lambda x: self.sim.schedule_event(latency, lambda: x.queue_message(m)))(n)
            (lambda x: self.sim.schedule_event(latency, lambda: x.proc_message(m, self.node_id)))(n)
        self.past_messages.add(m.id)

    def queue_message(self, m):
        self.message_queue.put_nowait(m)

    def generate_message(self):
        # pass onto all neighbors
        m = Message(self.node_id, self.node_id, "hello")
        self.send(m)

    def proc_message(self, m, source):
        # with probability p, passes message to all friends
        #print self.node_id, " received message from ", m.source
        self.send(m)
    
    # def loop(self):
    #     #print "node ", self.node_id, "processing loop"
    #     if not self.message_queue.empty():
    #         m = self.message_queue.get_nowait()
    #         self.proc_message(m, self.node_id)

    #     #print "Scheduling loop event: ", self.node_id
    #     self.sim.schedule_event(1, self.loop)

# honest node class
class HonestNode(Node):
    def __init__(self, node_id, sim):
        super(HonestNode, self).__init__(node_id, sim)
        
# compromised node class
class MaliciousNode(Node):
    def __init__(self, node_id, sim):
        super(MaliciousNode, self).__init__(node_id, sim)
        self.intercepted_messages = []
    
    def proc_message(self, m, source):
        self.intercepted_messages.append((self.sim.current_time, m.true_source, source, m.content))
        super(MaliciousNode, self).proc_message(m, source)
