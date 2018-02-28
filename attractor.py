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

    def __contains__(self, context):
        dimension = context.lattice.dimension()

        for i in range(0, len(self.contexts)):
            if context.lattice.issubset(self.contexts[i].lattice):
                return True
            if dimension > self.contexts[i].lattice.dimension():
                break

        return False

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


### Searching for attractors ###


class EventSet:
    def __init__(self, graph, index):
        self.graph = graph
        self.index = index
        self.configurations = []
        self.closures = []

    def copy(self):
        copy = EventSet(self.graph, self.index)
        copy.configurations = []
        copy.closures = list(self.closures)
        for config in self.configurations:
            copy.configurations.append(config.copy())

        return copy

    def branch(self, event):
        extension = self.copy()

        valid = False
        filtered = False

        for config in self.configurations:
            if event.preset.issubset(config.cut):
                valid = True
                extended_config = config.extend(event)
                if extended_config.parameter_context.empty():
                    continue
                new_index = marking_hash(extended_config.marking, self.graph)
                if new_index > self.index:
                    filtered = True
                    break
                extension.configurations.append(extended_config)

        if valid:
            self.closures.append(event)
            if filtered:
                return None
            else:
                return extension



def generate_all_markings(graph):
    markings = dict()

    partial_markings = [[]]

    for i in range(0, len(graph.nodes)):
        temporary_markings = []
        for m in partial_markings:
            for value in range(0, graph.nodes[i].maximum + 1):
                new_marking = list(m)
                new_marking.append(value)
                temporary_markings.append(new_marking)

        partial_markings = temporary_markings

    for m in partial_markings:
        markings[marking_hash(m, graph)] = m

    return markings


def compute_attractors(graph):
    all_markings = generate_all_markings(graph)

    handled_parametrisations = dict()
    attractors = []

    for index in all_markings:
        unfolding = parametrised_unfolding.init_unfolding(graph, all_markings[index])

        pe_queue = parametrised_unfolding.PossibleExtensionQueue()
        parametrised_unfolding.possible_extension(unfolding, unfolding.conditions[0], pe_queue)

        initial_config = Configuration()
        initial_config.cut = set(unfolding.conditions)
        initial_config.marking = list(unfolding.initial_marking)
        initial_config.parameter_context = parametrised_unfolding.ParameterContext(graph)

        trivial_attractor = EventSet(graph, index)
        trivial_attractor.configurations.append(initial_config)

        closed_event_sets = [trivial_attractor]

        event = unfolding.add_event(pe_queue, False)
        while event:
            tentative_event_sets = []

            for event_set in closed_event_sets:
                new_set = event_set.branch(event)

                if new_set:
                    tentative_event_sets.append(new_set)

            if len(tentative_event_sets):
                closed_event_sets.extend(tentative_event_sets)

                for cond in event.poset:
                    parametrised_unfolding.possible_extension(unfolding, cond, pe_queue)
            else:
                event.cutoff = True

            event = unfolding.add_event(pe_queue, False)

        for event_set in closed_event_sets:
            parameter_context = parametrised_unfolding.ParameterContext(graph)

            for config in event_set.configurations:
                parameter_context.intersect(config.parameter_context)
                if parameter_context.empty():
                    break

            for event in event_set.closures:
                if event.nature < 0:
                    parameter_context.limit_min(event.regulator_state, event.target_value + 1)
                elif event.nature > 0:
                    parameter_context.limit_max(event.regulator_state, event.target_value - 1)

            if parameter_context.empty():
                continue

            connected = True

            for config in event_set.configurations:
                reachable_index = marking_hash(config.marking, graph)
                if reachable_index in handled_parametrisations and parameter_context in handled_parametrisations[reachable_index]:
                    connected = False
                    break

            if connected:
                attractors.append(event_set)

                if index not in handled_parametrisations:
                    handled_parametrisations[index] = ParameterContextCollection()
                handled_parametrisations[index].add_context(parameter_context)

                print("Attractor: {0} with {1} for {2}-{3}.".format(
                    all_markings[index],
                    len(event_set.configurations),
                    parameter_context.lattice.min,
                    parameter_context.lattice.max
                ))


