import common
import numpy


minmax = False


class Node:
    def __init__(self):
        self.id = 0
        self.name = ""
        self.initial = False
        self.regulators = 0
        self.regulator_count = 0
        self.contexts = []
        self.maximum = 1

    def __str__(self):
        return self.name


class Edge:
    def __init__(self):
        self.id = 0
        self.source = Node()
        self.target = Node()
        self.observable = False
        self.monotonous = 0
        self.threshold = 0

    def __str__(self):
        return str(self.id) + ':' + str(self.source) + '->' + str(self.target)


class RegulatoryContext:
    def __init__(self, graph, target):
        self.id = 0
        self.target = target
        self.regulators = numpy.array([0] * len(graph.nodes))
        self.edges = [0] * len(graph.nodes)
        self.subcontexts = ([0] * len(graph.nodes))
        self.supercontexts = ([0] * len(graph.nodes))

    def extend(self, graph, edge):
        self.edges[edge.source.id] = edge
        ecs = []
        for val in range(1, (edge.source.maximum + 1)):
            ec = RegulatoryContext(graph, self.target)
            ec.id = graph.parametrisation_size
            ec.regulators = numpy.array(self.regulators)
            ec.regulators[edge.source.id] = val
            ec.edges = list(self.edges)
            ec.edges[edge.source.id] = edge

            if ecs:
                ecs[len(ecs) - 1].supercontexts[edge.source.id] = ec
                ec.subcontexts[edge.source.id] = ecs[len(ecs) - 1]
            else:
                self.supercontexts[edge.source.id] = ec
                ec.subcontexts[edge.source.id] = self

            for i in range(0, len(graph.nodes)):
                if (self.target.regulators & (1 << i)) and (not i == edge.source.id):
                    intermediary = ec.subcontexts[edge.source.id].subcontexts[i]
                    if intermediary and (intermediary.supercontexts[edge.source.id]):
                        intermediary.supercontexts[edge.source.id].supercontexts[i] = ec
                        ec.subcontexts[i] = intermediary.supercontexts[edge.source.id]

#            for e in self.edges:
#                intermediary = self.edges[e]
#                if edge in intermediary.edges:
#                    ec.edges[e] = intermediary.edges[edge]
#                    ec.edges[e].edges[e] = ec
#           ec.edges[edge] = self
#           self.edges[edge] = ec

            graph.contexts[ec.id] = ec
            graph.parametrisation_size += 1
            ecs.append(ec)
            if edge.threshold:
                ec.regulators[edge.source.id] = edge.source.maximum
                break
        return ecs

    def __str__(self):
        str = ''
        for r in self.regulators:
            str += ',' + str(r)
        str = str[1:]
        return '{' + str + '}'


class RegulatoryGraph:
    def __init__(self):
        self.parametrisation_size = 0
        self.nodes = []
        self.edges = []
        self.contexts = dict()
        self.known_parameters = dict()
        self.known_minimums = dict()
        self.known_maximums = dict()

    def get_node(self, node):
        if common.IsNumber(node):
            return self.nodes[int(node)]

        for n in self.nodes:
            if n.name == node:
                return n

    def build_marking(self, marking_string):
        marking = [0] * len(self.nodes)
        if common.IsNumber(marking_string):
            for i in range(0, len(self.nodes)):
                marking[i] = int(marking_string[i])
        else:
            nds = marking_string.split(",")
            for i in range(0, len(nds)):
                val = nds[i].split("=")
                node = self.get_node(val[0].strip())
                marking[node.id] = int(val[1])
        return marking



def parse_regulatory_graph(filename):
    prnfile = open(filename, 'r')

    graph = RegulatoryGraph()
    
    line = prnfile.readline()

    while not line.isspace():
        substrs = line.split(":") 
        values = substrs[1].split("/")

        node = Node()
        node.id = len(graph.nodes)
        node.name = substrs[0]
        node.initial = int(values[0])
        if len(values) == 2:
            node.maximum = int(values[1])
        graph.nodes.append(node)
        
        line = prnfile.readline()

    line = prnfile.readline()
    
    while line and (not line.isspace()):
        substrs = line.split(":")
        nds = substrs[0].split(";")
        src = nds[0].split(">")
        
        edge = Edge()
        edge.id = len(graph.edges)
        edge.source = graph.get_node(src[0].strip())
        edge.target = graph.get_node(nds[1].strip())

        if edge.target.regulator_count == 0:
            empty_con = RegulatoryContext(graph, edge.target)
            empty_con.id = graph.parametrisation_size
            graph.parametrisation_size += 1
            edge.target.contexts.append(empty_con)
            graph.contexts[empty_con.id] = empty_con

        edge.target.regulators += (1 << edge.source.id)
        edge.target.regulator_count += 1
        subcontexts = []
        for c in edge.target.contexts:
            subcontexts.append(c)

        if len(substrs) > 1:
            if '+' in substrs[1]:
                edge.monotonous = 1
            if '-' in substrs[1]:
                edge.monotonous = -1
            if 'o' in substrs[1]:
                edge.observable = True

        if len(src) > 1:
            edge.threshold = int(src[1])

        for c in subcontexts:
            ecs = c.extend(graph, edge)
            edge.target.contexts += ecs
        
        graph.edges.append(edge)
        
        line = prnfile.readline()

    line = prnfile.readline()

    for node in graph.nodes:
        if node.contexts:
            pos = minmax
            inf_regs = numpy.array([0] * len(graph.nodes))
            sup_regs = numpy.array([0] * len(graph.nodes))
            for edge in node.contexts[0].edges:
                if not edge:
                    continue
                if edge.monotonous:
                    if edge.monotonous > 0:
                        inf_regs[edge.source.id] = 0
                        sup_regs[edge.source.id] = edge.source.maximum
                    else:
                        inf_regs[edge.source.id] = edge.source.maximum
                        sup_regs[edge.source.id] = 0
                else:
                    pos = False
                    break
                if edge.observable:
                    pos |= True
            if pos:
                for c in node.contexts:
                    if (c.regulators == inf_regs).all():
                        graph.known_maximums[c.id] = (node.maximum - 1)
                        if minmax:
                            graph.known_maximums[c.id] = 0
                    if (c.regulators == sup_regs).all():
                        graph.known_minimums[c.id] = 1
                        if minmax:
                            graph.known_maximums[c.id] = node.maximum


    while line and (not line.isspace()):
        substrs = line.split('|')
        target = graph.get_node(substrs[0].strip())
        opstrs = substrs[1].split('=')
        mode = 0
        if len(opstrs) <= 1:
            opstrs = substrs[1].split('<')
            if len(opstrs) > 1:
                mode = -1
            else:
                opstrs = substrs[1].split('>')
                mode = 1

        regstr = opstrs[0].split(';')
        regulators = numpy.array([0] * len(graph.nodes))
        for r in regstr:
            if r and (not r.isspace()):
                regval = r.split(':')
                reg = graph.get_node(regval[0].strip())
                regulators[reg] = 1
                if len(regval) > 1:
                    regulators[reg] = int(regval[1])
        value = int(opstrs[1])

        for c in target.contexts:
            if (regulators == c.regulators).all():
                if not mode:
                    graph.known_parameters[c.id] = value
                elif mode < 0:
                    graph.known_minimums[c.id] = value
                else:
                    graph.known_maximums[c.id] = value
                break

        line = prnfile.readline()

    prnfile.close()
    return graph
