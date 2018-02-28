import parametrised_unfolding


class Configuration:
    def __init__(self):
        self.events = set()
        self.cut = set()
        self.marking = []
        self.parameter_context = 0

    def copy(self):
        copy = Configuration()
        copy.events = set(self.events)
        copy.cut = set(self.cut)
        copy.marking = list(self.marking)
        copy.parameter_context = self.parameter_context.copy()

        return copy

    def extend(self, event):
        new_config = self.copy()
        new_config.events.add(event)
        new_config.cut = self.cut.difference(event.preset).union(event.poset)
        new_config.marking[event.target.id] = event.target_value
        new_config.parameter_context.intersect(event.parameter_context)

        return new_config


class ParameterContextCollection:
    def __init__(self):
        self.contexts = []

    def copy(self):
        copy = ParameterContextCollection()
        for context in self.contexts:
            copy.add_context(context.copy())

        return copy

    def empty(self):
        for context in self.contexts:
            if not context.empty():
                return False

        return True

    def add_context(self, context):
        if context.empty():
            return

        dimension = context.lattice.dimension()
        found = False
        i = 0
        while i < len(self.contexts):
            cdim = self.contexts[i].lattice.dimension()
            if cdim > dimension:
                if context.lattice.issubset(self.contexts[i].lattice):
                    found = True
                    break
            elif cdim == dimension:
                dist = self.contexts[i].lattice.distance(context.lattice)
                if not dist:
                    found = True
                    break
                elif dist == 1:
                    union = self.contexts[i].union(context)
                    del self.contexts[i]
                    self.add_context(union)
                    found = True
                    break
            elif cdim < dimension:
                if not found:
                    found = True
                    self.contexts.insert(i, context)
                    i += 1
                if self.contexts[i].lattice.issubset(context.lattice):
                    del self.contexts[i]
                    i -= 1
            i += 1

        if not found:
            self.contexts.append(context)

    def intersection_with_single(self, context):
        intersection = ParameterContextCollection()
        for con in self.contexts:
            new_con = con.copy()
            new_con.intersect(context)
            intersection.add_context(new_con)

        return intersection

    def intersection(self, collection):
        intersection = ParameterContextCollection()
        for context in self.contexts:
            partial_section = collection.intersection_with_single(context)
            for con in partial_section.contexts:
                intersection.add_context(con)

        return intersection


def marking_hash(marking, graph):
    index = 0
    for n in graph.nodes:
        index *= (2 ** ((n.maximum // 2) + 1))
        index += marking[n.id]

    return index


def identify_attractor(graph, attractor):
    attractor_hash = set()
    for marking in attractor:
        attractor_hash.add(marking_hash(marking, graph))

    closure = parametrised_unfolding.ParameterContext(graph)

    for marking in attractor:
        for i in range(0,len(graph.nodes)):
            decrease = list(marking)
            found_decrease = False
            if marking[i] > 0:
                decrease[i] -= 1
                found_decrease = True

            increase = list(marking)
            found_increase = False
            if marking[i] < graph.nodes[i].maximum:
                increase[i] += 1
                found_increase = True

            for regulator_state in graph.nodes[i].regulator_states:
                match = True
                for j in range(0,len(graph.nodes)):
                    if (graph.nodes[i].regulators & (1 << j)) and (regulator_state.regulators[j] != marking[j]):
                        match = False
                        break
                if match:
                    break

            found_decrease &= marking_hash(decrease, graph) in attractor_hash
            found_increase &= marking_hash(increase, graph) in attractor_hash

            if not found_decrease:
                closure.limit_min(regulator_state, marking[i])
            if not found_increase:
                closure.limit_max(regulator_state, marking[i])

    connectivity_matrix = dict()

    for marking in attractor:
        source_hash = marking_hash(marking, graph)
        connectivity_matrix[source_hash] = dict()

        unfolding = parametrised_unfolding.init_unfolding(graph, marking, closure)

        pe_queue = parametrised_unfolding.PossibleExtensionQueue()
        parametrised_unfolding.possible_extension(unfolding, unfolding.conditions[0], pe_queue)

        configurations = []

        initial_config = Configuration()
        initial_config.cut = set(unfolding.conditions)
        initial_config.marking = list(unfolding.initial_marking)
        initial_config.parameter_context = closure.copy()
        configurations.append(initial_config)

        while True:
            tentative_configurations = []

            event = unfolding.add_event(pe_queue)

            if not event:
                break

            for config in configurations:
                if config.cut.issuperset(event.preset):
                    tentative_configurations.append(config.extend(event))

            configurations.extend(tentative_configurations)

        for config in configurations:
            target_hash = marking_hash(config.marking, graph)
            if not target_hash in connectivity_matrix[source_hash]:
                connectivity_matrix[source_hash][target_hash] = ParameterContextCollection()

            connectivity_matrix[source_hash][target_hash].add_context(config.parameter_context)

    parametrisations = ParameterContextCollection()
    parametrisations.add_context(closure)

    for source in attractor:
        source_hash = marking_hash(source, graph)
        for target in attractor:
            target_hash = marking_hash(target, graph)
            if target_hash in connectivity_matrix[source_hash]:
                parametrisations = parametrisations.intersection(connectivity_matrix[source_hash][target_hash])
            else:
                parametrisations.contexts.clear()

    return parametrisations