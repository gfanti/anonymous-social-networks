import Queue
import networkx as nx
import random
import math
import numpy as np

class NetworkLatency(object):
    @classmethod
    def next(cls, p):
        return np.random.geometric(p, 1)[0]

class Message(object):
    def __init__(self, source, content):
        self.id = random.getrandbits(64)
        self.content = content
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
            # schedule 
            (lambda x: self.sim.schedule_event(latency, lambda: x.queue_message(m)))(n)
        self.past_messages.add(m.id)

    def queue_message(self, m):
        self.message_queue.put_nowait(m)

    def generate_message(self):
        # pass onto all neighbors
        m = Message(self.node_id, "hello")
        self.send(m)

    def proc_message(self, m):
        # with probability p, passes message to all friends
        self.send(m)
    
    def loop(self):
        if not self.message_queue.empty():
            m = self.message_queue.get_nowait()
            self.proc_message(m)

        #print "Scheduling loop event: ", self.node_id
        self.sim.schedule_event(2, self.loop)

# honest node class
class HonestNode(Node):
    def __init__(self, node_id, sim):
        super(HonestNode, self).__init__(node_id, sim)
        
# compromised node class
class MaliciousNode(Node):
    def __init__(self, node_id, sim):
        super(MaliciousNode, self).__init__(node_id, sim)
        self.intercepted_messages = []
    
    def proc_message(self, m):
        super(MaliciousNode, self).proc_message(m)
        self.intercepted_messages.append((self.sim.current_time, m.source, m.content))
