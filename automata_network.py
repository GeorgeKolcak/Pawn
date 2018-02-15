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

        for node in self.graph.nodes:
            if not self.mask & (1 << node.id):
                continue
            for i in range(0, regulator_state.target.maximum + 1):
                clean_index = index - (regulator_state.regulators[node.id] << (self.index_length - self.index_map[node.id]))
                complete_index = clean_index + (i << (self.index_length - self.index_map[node.id]))
                if self.clauses[complete_index] is None:
                    self.clauses[complete_index] = Clause()
                if i == regulator_state.regulators[node.id]:
                    continue
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
                if clause is not None and clause.regulator_state is not None and node.id in clause.siblings\
                        and len(clause.siblings[node.id]) == node.maximum:
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

                if self.mask:
                    regulator_string = ""
                    for node in self.graph.nodes:
                        if not self.mask & (1 << node.id):
                            continue
                        regulator_string += " and \"{0}\"={1}".format(node.name, clause.regulator_state.regulators[node.id])
                    regulator_string = regulator_string[5:]

                    transition_string += " when {0}\n".format(regulator_string)

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
    def __init__(self, graph, event):
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
                    if (not node.regulators & (1 << node.id) or regulator_state.regulators[node.id] == i + 1)\
                            and event.parameter_context.lattice.min[regulator_state.id] < i + 1:
                        inhibitions.add_regulator_state(regulator_state)
                    if (not node.regulators & (1 << node.id) or regulator_state.regulators[node.id] == i)\
                            and event.parameter_context.lattice.max[regulator_state.id] > i:
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
            marking_string += ", \"{0}\"={1}".format(node.name, event.marking[node.id])
        marking_string = marking_string[2:]

        data += "initial_context {0}\n".format(marking_string)

        super().__init__(data.encode(encoding='UTF-8'))

    def populate_popen_args(self, args, kwargs): #workaround until pypint is fixed
        kwargs["input"] = self.data
        super(pypint.InMemoryModel, self).populate_popen_args(args, kwargs)
