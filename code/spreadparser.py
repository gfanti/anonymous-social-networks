# parsing.py
import estimation

class Parser(object):

    def __init__(self, filename):
        self.filename = filename
        
    def parse_file(self):
        # parses an output file from the spread of the message
        # Outputs:
        #   adjacency: an adjacency matrix for the underlying graph
        #   source: a list of sets with the underlying graph
        #   malicious_nodes: a list of nodes that have been deemed 'malicious' 
        #   timestamps: a list of times at which each malicious node received the message
        
        f = open(self.filename, 'r')
        
        adjacency = []
        malicious_nodes = []
        while True:
            line = f.readline()
            line = line.split()
            if line[0] == 'Simulation':
                break
            id = int(line[0])
            if line[1] == 'bad':
                malicious_nodes.append(int(line[0]))
            neighbors = f.readline()
            neighbors = neighbors.strip('[]\n').split(',')
            neighbors = [int(i) for i in neighbors]
            # print('neighbors are : ', neighbors)
            adjacency.append(set(neighbors))
            
        # Now parse the times at which malicious nodes got the message
        timestamps = []
        infectors = []
        while True:
            line = f.readline()
            if not line:
                break
            line = line.split()
            line = [item.strip('[(,)]') for item in line]
            timestamps.append(float(line[1]))
            source = int(line[2])
            infectors.append(int(line[3]))
        return source, adjacency, malicious_nodes, timestamps, infectors

class MultiMessageParser(object):

    def __init__(self, filename):
        self.filename = filename
        
    def parse_file(self):
        # parses an output file from the spread of the message
        # Outputs:
        #   adjacency: an adjacency matrix for the underlying graph
        #   source: a list of sets with the underlying graph
        #   malicious_nodes: a list of nodes that have been deemed 'malicious' 
        #   timestamps: a list of times at which each malicious node received the message
        
        f = open(self.filename, 'r')
        
        adjacency = []
        malicious_nodes = []
        while True:
            line = f.readline()
            line = line.split()
            if line[0] == 'Simulation':
                break
            id = int(line[0])
            if line[1] == 'bad':
                malicious_nodes.append(int(line[0]))
            neighbors = f.readline()
            neighbors = neighbors.strip('[]\n').split(',')
            neighbors = [int(i) for i in neighbors]
            # print('neighbors are : ', neighbors)
            adjacency.append(set(neighbors))
            
        # Now parse the times at which malicious nodes got the message
        timestamps = [[]]
        while True:
            line = f.readline()
            if not line:
                break
            line = line.split()
            if line[0] == 'Simulation':
                timestamps.append([])
                continue
            line = [item.strip('[(,)]') for item in line]
            timestamps[-1].append(float(line[1]))
            source = int(line[2])
        
        return source, adjacency, malicious_nodes, timestamps
