import pypint


class Clause:
    def __init__(self):
        self.regulator_state = None
        self.siblings = dict()
        self.is_subsumed = False

    def add_sibling(self, node, sibling):
        if node.id not in self.siblings:
            self.siblings[node.id] = set()

        self.siblings[node.id].add(sibling)


class FormulaBuilder:
    def __init__(self, graph, mask):
        self.graph = graph
        self.mask = mask
        self.clauses = [None]
        self.index_map = dict()

        last_index = 0
        for node in graph.nodes:
            if not mask & (1 << node.id):
                continue
            self.index_map[node.id] = last_index
            for i in range(0, (node.maximum // 2) + 1):
                self.clauses += self.clauses

            last_index += (node.maximum // 2) + 1

        self.index_length = last_index - 1

    def empty(self):
        for clause in self.clauses:
            if clause is not None and clause.regulator_state is not None:
                return False

        return True

    def add_regulator_state(self, regulator_state):
        index = 0
        for node in self.graph.nodes:
            if not self.mask & (1 << node.id):
                continue
            index *= (2 ** ((node.maximum // 2) + 1))
            index += regulator_state.regulators[node.id]

        if self.clauses[index] is None:
            self.clauses[index] = Clause()
        if self.clauses[index].regulator_state is not None:
            return

        self.clauses[index].regulator_state = regulator_state

        if not regulator_state_monotonically_minimal(regulator_state):
            for node in self.graph.nodes:
                if not self.mask & (1 << node.id):
                    continue
                clean_index = index - (regulator_state.regulators[node.id] << (self.index_length - self.index_map[node.id]))
                for i in range(0, regulator_state.target.maximum + 1):
                    if i == regulator_state.regulators[node.id]:
                        continue

                    complete_index = clean_index + (i << (self.index_length - self.index_map[node.id]))
                    if self.clauses[complete_index] is None:
                        self.clauses[complete_index] = Clause()
                    else:
                        self.clauses[complete_index].add_sibling(node, complete_index)

    def collapse(self):
        new_builders = []

        for node in self.graph.nodes:
            if not self.mask & (1 << node.id):
                continue
            new_mask = self.mask - (1 << node.id)
            new_layer = FormulaBuilder(self.graph, new_mask)

            for clause in self.clauses:
                if clause is not None and clause.regulator_state is not None and node.id in clause.siblings and \
                        len(clause.siblings[node.id]) == node.maximum and clause.regulator_state.edges[node.id].monotonous and \
                        not regulator_state_monotonically_minimal(clause.regulator_state):
                    new_layer.add_regulator_state(clause.regulator_state)
                    clause.is_subsumed = True
                    for sibling in clause.siblings[node.id]:
                        self.clauses[sibling].is_subsumed = True

            if not new_layer.empty():
                new_builders.append(new_layer)

        return new_builders

    def extract_transitions(self):
        transitions = []

        for clause in self.clauses:
            if clause is not None and clause.regulator_state is not None and not clause.is_subsumed:
                transition_string = "\"{0}\" ".format(clause.regulator_state.target.name)
                transition_string += "{0}"

                monotonically_minimal = regulator_state_monotonically_minimal(clause.regulator_state)

                regulator_string = ""
                for node in self.graph.nodes:
                    if not self.mask & (1 << node.id) or (monotonically_minimal and clause.regulator_state.edges[node.id].monotonous):
                        continue
                    regulator_string += " and \"{0}\"={1}".format(node.name, clause.regulator_state.regulators[node.id])
                regulator_string = regulator_string[5:]

                if regulator_string:
                    transition_string += " when {0}\n".format(regulator_string)
                else:
                    transition_string += "\n"

                transitions.append(transition_string)

        return transitions

    def merge(self, builder):
        if builder.mask != self.mask:
            return

        for i in range(0, len(self.clauses)):
            if self.clauses[i] is None:
                if builder.clauses[i] is not None:
                    self.clauses[i] = builder.clauses[i]
            else:
                if self.clauses[i].regulator_state is None:
                    self.clauses[i].regulator_state = builder.clauses[i].regulator_state
                for sibling in builder.clauses[i].siblings:
                    if sibling in self.clauses[i].siblings:
                        self.clauses[i].siblings[sibling] = self.clauses[i].siblings[sibling].union(builder.clauses[i].siblings[sibling])
                    else:
                        self.clauses[i].siblings[sibling] = builder.clauses[i].siblings[sibling]


class BuilderCollection():
    def __init__(self):
        self.builders = dict()

    def __len__(self):
        return len(self.builders)

    def add(self, builder):
        if builder.mask in self.builders:
            self.builders[builder.mask].merge(builder)
        else:
            self.builders[builder.mask] = builder

    def add_many(self, builders):
        for builder in builders:
            self.add(builder)


class ConfigurationWrapperModel(pypint.InMemoryModel):
    def __init__(self, graph, context, marking):
        data = ""

        for node in graph.nodes:
            data += "\"{0}\" {1}\n".format(node.name, list(range(0, node.maximum + 1)))

        for node in graph.nodes:
            for i in range(0, node.maximum):
                mask = node.regulators
                if node.regulators & (1 << node.id):
                    mask -= (1 << node.id)

                inhibitions = FormulaBuilder(graph, mask)
                activations = FormulaBuilder(graph, mask)

                for regulator_state in node.regulator_states:

                    if (not node.regulators & (1 << node.id) or regulator_state.regulators[node.id] == i + 1) \
                            and context.lattice.min[regulator_state.id] < i + 1 \
                            and context.allowed_lattice.min[regulator_state.id] < i + 1:
                        inhibitions.add_regulator_state(regulator_state)
                    if (not node.regulators & (1 << node.id) or regulator_state.regulators[node.id] == i)\
                            and context.lattice.max[regulator_state.id] > i \
                            and context.allowed_lattice.max[regulator_state.id] > i:
                        activations.add_regulator_state(regulator_state)

                inhibition_string = "{0} -> {1}".format(i + 1, i)
                activation_string = "{0} -> {1}".format(i, i + 1)

                inhibition_refinements = BuilderCollection()
                inhibition_refinements.add_many(inhibitions.collapse())
                for transition in inhibitions.extract_transitions():
                    data += transition.format(inhibition_string)

                activation_refinements = BuilderCollection()
                activation_refinements.add_many(activations.collapse())
                for transition in activations.extract_transitions():
                    data += transition.format(activation_string)

                while len(inhibition_refinements):
                    new_refinements = BuilderCollection()
                    for refinement in inhibition_refinements.builders.values():
                        new_refinements.add_many(refinement.collapse())
                        for transition in refinement.extract_transitions():
                            data += transition.format(inhibition_string)

                    inhibition_refinements = new_refinements

                while len(activation_refinements):
                    new_refinements = BuilderCollection()
                    for refinement in activation_refinements.builders.values():
                        new_refinements.add_many(refinement.collapse())
                        for transition in refinement.extract_transitions():
                            data += transition.format(activation_string)

                    activation_refinements = new_refinements

        marking_string = ""
        for node in graph.nodes:
            marking_string += ", \"{0}\"={1}".format(node.name, marking[node.id])
        marking_string = marking_string[2:]

        data += "initial_context {0}\n".format(marking_string)

        super().__init__(data.encode(encoding='UTF-8'))

    def populate_popen_args(self, args, kwargs): #workaround until pypint is fixed
        kwargs["input"] = self.data
        super(pypint.InMemoryModel, self).populate_popen_args(args, kwargs)


def regulator_state_monotonically_minimal(regulator_state):
    for edge in regulator_state.edges:
        if edge is None:
            continue
        if (edge.monotonous == 1 and regulator_state.regulators[edge.source.id] != 0) or \
                (edge.monotonous == -1 and regulator_state.regulators[edge.source.id] != edge.source.maximum):
            return False

    return True


class Transition:
    def __init__(self, regulator_state, regulator_mask):
        self.regulator_state = regulator_state
        self.regulator_mask = regulator_mask

    def __contains__(self, regulator_state):
        mask = self.regulator_mask & regulator_state.target.regulators
        for i in range(0, len(self.regulator_state)):
            if mask & (1 << i) and self.regulator_state[i] != regulator_state.regulators[i]:
                return False

        return True


class TransitionCollection:
    def __init__(self, graph):
        self.graph = graph
        self.transitions = []

    def add_transition(self, an_transition):
        regulator_mask = 0
        regulator_state = [0] * len(self.graph.nodes)
        for condition in an_transition.conds:
            regulator = self.graph.get_node(condition)
            regulator_mask += (1 << regulator.id)
            regulator_state[regulator.id] = an_transition.conds[condition]

        target = self.graph.get_node(an_transition.a)
        if target.regulators & (1 << target.id):
            regulator_mask += (1 << target.id)
            regulator_state[target.id] = an_transition.i

        self.transitions.append(Transition(regulator_state, regulator_mask))

    def __contains__(self, regulator_state):
        for transition in self.transitions:
            if regulator_state in transition:
                return True

        return False


def restrict_context_to_model(graph, context, model):
    activations = [None] * len(graph.nodes)
    inhibitions = [None] * len(graph.nodes)

    for node in graph.nodes:
        activations[node.id] = [None] * node.maximum
        inhibitions[node.id] = [None] * node.maximum

        for i in range(0, node.maximum):
            activations[node.id][i] = TransitionCollection(graph)
            inhibitions[node.id][i] = TransitionCollection(graph)

    for transition in model.local_transitions:
        target = graph.get_node(transition.a)

        if transition.i < transition.j:
            activations[target.id][transition.i].add_transition(transition)
        else:
            inhibitions[target.id][transition.j].add_transition(transition)

    for node in graph.nodes:
        for regulator_state in node.regulator_states:
            induced_maximum = None
            induced_minimum = None

            for i in range(0, node.maximum):
                if regulator_state in activations[node.id][i]:
                    if induced_maximum is None:
                        induced_maximum = i + 1
                    else:
                        induced_maximum = max(induced_maximum, i + 1)

                if regulator_state in inhibitions[node.id][i]:
                    if induced_minimum is None:
                        induced_minimum = i
                    else:
                        induced_minimum = min(induced_minimum, i)

            if induced_maximum is None:
                context.forbid_activation(regulator_state)
            else:
                context.soft_limit_max(regulator_state, induced_maximum)

            if induced_minimum is None:
                context.forbid_inhibition(regulator_state)
            else:
                context.soft_limit_min(regulator_state, induced_minimum)

