import common
import numpy


minmax = False


class Node:
    def __init__(self):
        self.id = 0
        self.name = ""
        self.initial = 0
        self.regulators = 0
        self.regulator_count = 0
        self.regulator_states = []
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


class RegulatorState:
    def __init__(self, graph, target):
        self.id = 0
        self.target = target
        self.regulators = numpy.array([0] * len(graph.nodes))
        self.edges = [0] * len(graph.nodes)
        self.substates = ([0] * len(graph.nodes))
        self.superstates = ([0] * len(graph.nodes))

    def extend(self, graph, edge):
        self.edges[edge.source.id] = edge
        extended_states = []
        for val in range(1, (edge.source.maximum + 1)):
            extended_state = RegulatorState(graph, self.target)
            extended_state.id = graph.parametrisation_size
            extended_state.regulators = numpy.array(self.regulators)
            extended_state.regulators[edge.source.id] = val
            extended_state.edges = list(self.edges)
            extended_state.edges[edge.source.id] = edge

            if extended_states:
                extended_states[len(extended_states) - 1].superstates[edge.source.id] = extended_state
                extended_state.substates[edge.source.id] = extended_states[len(extended_states) - 1]
            else:
                self.superstates[edge.source.id] = extended_state
                extended_state.substates[edge.source.id] = self

            for i in range(0, len(graph.nodes)):
                if (self.target.regulators & (1 << i)) and (not i == edge.source.id):
                    intermediary = extended_state.substates[edge.source.id].substates[i]
                    if intermediary and (intermediary.superstates[edge.source.id]):
                        intermediary.superstates[edge.source.id].superstates[i] = extended_state
                        extended_state.substates[i] = intermediary.superstates[edge.source.id]

            graph.regulator_states[extended_state.id] = extended_state
            graph.parametrisation_size += 1
            extended_states.append(extended_state)
            if edge.threshold:
                extended_state.regulators[edge.source.id] = edge.source.maximum
                break
        return extended_states

    def __str__(self):
        string = ''
        for r in self.regulators:
            string += ',' + string(r)

        string = string[1:]
        return '{' + string + '}'


class PartialState:
    def __init__(self, graph):
        self.graph = graph
        self.mask = numpy.array([False] * len(graph.nodes))
        self.values = numpy.array([0] * len(graph.nodes))

    def matches(self, marking):
        return ((marking == self.values) | numpy.invert(self.mask)).all()

    def __str__(self):
        string = ""

        for node in self.graph.nodes:
            if self.mask[node.id]:
                string += ",{0}={1}".format(node.name, self.values[node.id])

        return string[1:]


class RegulatoryGraph:
    def __init__(self):
        self.parametrisation_size = 0
        self.nodes = []
        self.edges = []
        self.regulator_states = dict()
        self.known_parameters = dict()
        self.known_minimums = dict()
        self.known_maximums = dict()

    def copy(self):
        new_graph = RegulatoryGraph()
        new_graph.parametrisation_size = self.parametrisation_size
        new_graph.nodes = list(self.nodes)
        new_graph.edges = list(self.nodes)
        new_graph.regulator_states = dict(self.regulator_states)
        new_graph.known_parameters = dict(self.known_parameters)
        new_graph.known_minimums = dict(self.known_minimums)
        new_graph.known_maximums = dict(self.known_maximums)

        return new_graph

    def get_node(self, node):
        if common.is_number(node):
            return self.nodes[int(node)]

        for n in self.nodes:
            if n.name == node:
                return n


def parse_regulatory_graph(filename):
    prn_file = open(filename, 'r')

    graph = RegulatoryGraph()
    
    line = prn_file.readline()

    while not line.isspace():
        substrings = line.split(":")
        values = substrings[1].split("/")

        node = Node()
        node.id = len(graph.nodes)
        node.name = substrings[0]
        node.initial = int(values[0])
        if len(values) == 2:
            node.maximum = int(values[1])
        graph.nodes.append(node)
        
        line = prn_file.readline()

    line = prn_file.readline()
    
    while line and (not line.isspace()):
        substrings = line.split(":")
        node_strings = substrings[0].split(";")
        source = node_strings[0].split(">")
        
        edge = Edge()
        edge.id = len(graph.edges)
        edge.source = graph.get_node(source[0].strip())
        edge.target = graph.get_node(node_strings[1].strip())

        if edge.target.regulator_count == 0:
            empty_regulator_state = RegulatorState(graph, edge.target)
            empty_regulator_state.id = graph.parametrisation_size
            graph.parametrisation_size += 1
            edge.target.regulator_states.append(empty_regulator_state)
            graph.regulator_states[empty_regulator_state.id] = empty_regulator_state

        edge.target.regulators += (1 << edge.source.id)
        edge.target.regulator_count += 1
        substates = list(edge.target.regulator_states)

        if len(substrings) > 1:
            if '+' in substrings[1]:
                edge.monotonous = 1
            if '-' in substrings[1]:
                edge.monotonous = -1
            if 'o' in substrings[1]:
                edge.observable = True

        if len(source) > 1:
            edge.threshold = int(source[1])

        for regulator_state in substates:
            extended_regulator_state = regulator_state.extend(graph, edge)
            edge.target.regulator_states += extended_regulator_state
        
        graph.edges.append(edge)
        
        line = prn_file.readline()

    line = prn_file.readline()

    for node in graph.nodes:
        if node.regulator_states:
            possible = minmax
            inhibitors = numpy.array([0] * len(graph.nodes))
            activators = numpy.array([0] * len(graph.nodes))
            for edge in node.regulator_states[0].edges:
                if not edge:
                    continue
                if edge.monotonous:
                    if edge.monotonous > 0:
                        inhibitors[edge.source.id] = 0
                        activators[edge.source.id] = edge.source.maximum
                    else:
                        inhibitors[edge.source.id] = edge.source.maximum
                        activators[edge.source.id] = 0
                else:
                    possible = False
                    break
                if edge.observable:
                    possible |= True
            if possible:
                for regulator_state in node.regulator_states:
                    if (regulator_state.regulators == inhibitors).all():
                        if minmax:
                            graph.known_maximums[regulator_state.id] = 0
                        else:
                            graph.known_maximums[regulator_state.id] = (node.maximum - 1)
                    if (regulator_state.regulators == activators).all():
                        if minmax:
                            graph.known_minimums[regulator_state.id] = node.maximum
                        else:
                            graph.known_minimums[regulator_state.id] = 1

    while line and (not line.isspace()):
        substrings = line.split('|')
        target = graph.get_node(substrings[0].strip())
        operator_strings = substrings[1].split('=')
        mode = 0
        if len(operator_strings) <= 1:
            operator_strings = substrings[1].split('<')
            if len(operator_strings) > 1:
                mode = -1
            else:
                operator_strings = substrings[1].split('>')
                mode = 1

        regulator_strings = operator_strings[0].split(';')
        regulators = numpy.array([0] * len(graph.nodes))
        for regulator_string in regulator_strings:
            if regulator_string and (not regulator_string.isspace()):
                regulator_value_strings = regulator_string.split(':')
                regulator = graph.get_node(regulator_value_strings[0].strip())
                regulators[regulator.id] = 1
                if len(regulator_value_strings) > 1:
                    regulators[regulator.id] = int(regulator_value_strings[1])
        value = int(operator_strings[1])

        for regulator_state in target.regulator_states:
            if (regulators == regulator_state.regulators).all():
                if not mode:
                    graph.known_parameters[regulator_state.id] = value
                elif mode > 0:
                    graph.known_minimums[regulator_state.id] = value
                else:
                    graph.known_maximums[regulator_state.id] = value
                break

        line = prn_file.readline()

    prn_file.close()
    return graph
