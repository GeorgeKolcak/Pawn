import pypint


class RegulationLimitCollection:
    def __init__(self, nature):
        self.nature = nature
        self.limits = []

    def add_limit(self, regulator_state):
        i = 0
        while i < len(self.limits):
            comparison = regulator_state.monotonic_compare(self.limits[i])

            if comparison * self.nature < 0:
                return
            elif comparison * self.nature > 0:
                self.limits.remove(self.limits[i])
                i -= 1

            i += 1

        self.limits.append(regulator_state)


class PartialRegulatorStateHierarchy:
    def __init__(self, graph, condition, nature):
        self.graph = graph
        self.condition = condition
        self.nature = nature

        self.layers = {}

    def compute_layers(self):
        mask_layers = self._generate_masks()

        for i in range(0, self.condition.node.regulator_count):
            self.layers[i] = self._generate_layer(mask_layers[i])

    def copy(self):
        copy = PartialRegulatorStateHierarchy(self.graph, self.condition, self.nature)

        for i in range(0, self.condition.node.regulator_count):
            copy.layers[i] = list(self.layers[i])

    def _generate_layer(self, masks):
        layer = []

        for mask in masks:
            for regulator_state in self.condition.node.regulator_states:
                for i in range(0, len(self.graph.nodes)):
                    if (mask & 1 << i) and regulator_state.regulators[i] != 0:
                        continue

                    layer.append(PartialRegulatorState(regulator_state, mask))

        return layer

    def _generate_masks(self):
        regulators = []

        for i in range(0, len(self.graph.nodes)):
            if self.condition.node.regulators & 1 << i:
                regulators.append(self.graph.nodes[i])

        mask_layers = {0: [0]}
        for i in range(1, len(regulators)):
            mask_layers[i] = []

            for mask in mask_layers[i - 1]:
                for regulator in regulators:
                    if not (mask & 1 << regulator.id) and (mask > regulator.id):
                        mask_layers[i].append(mask + 1 << regulator.id)

        return mask_layers

    def partial_regulator_states(self, layer):
        return self.layers[layer]


class PartialRegulatorState:
    def __init__(self, regulator_state, mask):
        self.regulator_state = regulator_state
        self.mask = mask

    def copy(self):
        copy = PartialRegulatorState(self.regulator_state, self.mask)

        return copy

    def matches(self, regulator_state):
        for i in range(0, len(regulator_state.regulators)):
            if (not (self.mask & 1 << i)) and regulator_state.regulators[i] != self.regulator_state.regulators[i]:
                return False

        return True


class RegulationPattern:
    def __init__(self, graph, nature, activator_mask, inhibitor_mask):
        self.graph = graph
        self.nature = nature
        self.activator_mask = activator_mask
        self.inhibitor_mask = inhibitor_mask
        self.mask = activator_mask + inhibitor_mask
        self.values = [None] * len(self.graph.nodes)

    @staticmethod
    def from_value(graph, nature, activator_mask, inhibitor_mask, regulator, value):
        pattern = RegulationPattern(graph, nature, activator_mask, inhibitor_mask)
        pattern.set_value(regulator, value)

        return pattern

    @staticmethod
    def from_regulator_state(graph, nature, activator_mask, inhibitor_mask, regulator_state):
        pattern = RegulationPattern(graph, nature, activator_mask, inhibitor_mask)
        for regulator in graph.nodes:
            if pattern.mask & (1 << regulator.id):
                pattern.set_value(regulator, regulator_state.regulators[regulator.id])

        return pattern

    def _copy(self):
        copy = RegulationPattern(self.graph, self.nature, self.activator_mask, self.inhibitor_mask)
        copy.values = list(self.values)

        return copy

    def _invert_values(self, regulator):
        return (self.activator_mask & (1 << regulator.id) and self.nature > 0) or \
                    (self.inhibitor_mask & (1 << regulator.id) and self.nature < 0)

    def set_value(self, regulator, value):
        self.values[regulator.id] = value

        if self._invert_values(regulator):
            self.values[regulator.id] = regulator.maximum - self.values[regulator.id]

        if self.values[regulator.id] == regulator.maximum:
            self.values[regulator.id] = None

    def superpattern(self, regulator):
        pattern = self._copy()

        pattern.values[regulator.id] = pattern.values[regulator.id] + 1

        return pattern

    def concrete_subpattern(self, regulator, value):
        pattern = self._copy()

        pattern.values[regulator.id] = value

        return pattern

    def __contains__(self, pattern):
        if self.mask != pattern.mask:
            return False

        for i in range(0, len(self.values)):
            if not self.mask & (1 << i):
                continue

            if pattern.values[i] is None:
                if self.values[i] is not None:
                    return False
            else:
                if self.values[i] is not None and self.values[i] < pattern.values[i]:
                    return False

        return True

    def __str__(self):
        constraints = ""

        for regulator in self.graph.nodes:
            if (not self.mask & (1 << regulator.id)) or self.values[regulator.id] is None:
                continue

            value = self.values[regulator.id]
            if self._invert_values(regulator):
                value = regulator.maximum - value

            constraints = constraints + " and \"{0}\"={1}".format(regulator.name, value)

        if len(constraints):
            return constraints[5:]

        return constraints


class RegulatorLattice:
    def __init__(self, graph, allowed_values, activator_mask, inhibitor_mask):
        self.graph = graph
        self.allowed_values = allowed_values
        self.activator_mask = activator_mask
        self.inhibitor_mask = inhibitor_mask
        self.mask = activator_mask + inhibitor_mask

        self.neutral_regulators = PartialRegulatorState()

        self.activation_patterns = []
        self.inhibition_patterns = []

    def initialise_patterns(self):
        for regulator in self.graph.nodes:
            is_activator = self.activator_mask & (1 << regulator.id)
            is_inhibitor = self.inhibitor_mask & (1 << regulator.id)
            if not (is_activator or is_inhibitor):
                continue

            for value in range(0, regulator.maximum + 1):
                if (is_activator and value > 0) or (is_inhibitor and value < regulator.maximum):
                    self.activation_patterns.append(RegulationPattern.from_value(self.graph, 1, self.activator_mask, self.inhibitor_mask, regulator, value))
                if (is_activator and value < regulator.maximum) or (is_inhibitor and value > 0):
                    self.inhibition_patterns.append(RegulationPattern.from_value(self.graph, -1, self.activator_mask, self.inhibitor_mask, regulator, value))

        if not len(self.activation_patterns):
            self.activation_patterns.append(RegulationPattern(self.graph, 1, self.activator_mask, self.inhibitor_mask))
        if not len(self.inhibition_patterns):
            self.inhibition_patterns.append(RegulationPattern(self.graph, 1, self.activator_mask, self.inhibitor_mask))

    def copy(self):
        copy = RegulatorLattice(self.graph, self.allowed_values, self.activator_mask, self.inhibitor_mask)
        copy.activation_patterns = list(self.activation_patterns)
        copy.inhibition_patterns = list(self.inhibition_patterns)
        copy.neutral_regulators = self.neutral_regulators.copy()

        return copy

    def add_neutral_regulator(self, regulator, value):
        extension = self.copy()
        extension.neutral_regulators = self.neutral_regulators.extend(regulator, value)

        return extension

    @staticmethod
    def _filter_subsumed_patterns(pattern_list, new_patterns):
        i = 0
        while i < len(pattern_list):
            for pattern in new_patterns:
                if pattern_list[i] in pattern:
                    pattern_list.remove(pattern_list[i])
                    i -= 1
                    break

            i += 1

    def _compute_superpatterns(self, seed):
        superpatterns = [seed]

        for regulator in self.graph.nodes:
            if (not self.mask & (1 << regulator.id)) or seed.values[regulator.id] is None:
                continue

            new_superpatterns = []

            for value in range(seed.values[regulator.id], regulator.maximum):
                for pattern in superpatterns:
                    new_superpatterns.append(pattern.superpattern(regulator))

            superpatterns = new_superpatterns

        return superpatterns

    def limit_inhibition(self, limit):
        limit_pattern = RegulationPattern.from_regulator_state(self.graph, -1, self.activator_mask, self.inhibitor_mask, limit)

        superpatterns = self._compute_superpatterns(limit_pattern)

        self._filter_subsumed_patterns(self.inhibition_patterns, superpatterns)

        for pattern in superpatterns:
            for regulator in self.graph.nodes:
                if (not self.mask & (1 << regulator.id)) or pattern.values[regulator.id] is not None:
                    continue

                for value in range(0, regulator.maximum):
                    self.inhibition_patterns.append(pattern.concrete_subpattern(regulator, value))

    def limit_activation(self, limit):
        limit_pattern = RegulationPattern.from_regulator_state(self.graph, 1, self.activator_mask, self.inhibitor_mask, limit)

        superpatterns = self._compute_superpatterns(limit_pattern)

        self._filter_subsumed_patterns(self.activation_patterns, superpatterns)

        for pattern in superpatterns:
            for regulator in self.graph.nodes:
                if (not self.mask & (1 << regulator.id)) or pattern.values[regulator.id] is not None:
                    continue

                for value in range(0, regulator.maximum):
                    self.activation_patterns.append(pattern.concrete_subpattern(regulator, value))

    def constraints(self, pattern):
        constraints = str(pattern)

        for i in range(0, len(self.neutral_regulators.regulators)):
            constraints = constraints + " and \"{0}\"={1}".format(self.neutral_regulators.regulators[i].name, self.neutral_regulators.values[i])

        if len(constraints):
            constraints = "when {0}".format(constraints)

        return constraints


class ConfigurationWrapperModel(pypint.InMemoryModel):
    def __init__(self, graph, context, marking):
        data = ""

        for node in graph.nodes:
            data += "\"{0}\" {1}\n".format(node.name, list(range(0, node.maximum + 1)))

        for node in graph.nodes:
            for value in range(0, node.maximum):
                can_activate = context.allowed_values.max[node.id] > value
                can_inhibit = 0 <= context.allowed_values.min[node.id] <= value

                if not (can_activate or can_inhibit):
                    continue

                self_regulated = node.regulators & (1 << node.id)

                activation_limits = RegulationLimitCollection(1)
                inhibition_limits = RegulationLimitCollection(-1)

                for regulator_state in node.regulator_states:
                    if can_inhibit and context.lattice.min[regulator_state.id] > value and \
                            (not self_regulated or regulator_state.regulators[node.id] == value + 1):
                        activation_limits.add_limit(regulator_state)

                    if can_activate and context.lattice.max[regulator_state.id] <= value and \
                            (not self_regulated or regulator_state.regulators[node.id] == value):
                        inhibition_limits.add_limit(regulator_state)

                activator_mask = 0
                inhibitor_mask = 0

                neutral_regulators = []

                for regulator in graph.nodes:
                    if not node.regulators & (1 << regulator.id) or regulator.id == node.id:
                        continue

                    if node.regulator_states[0].edges[regulator.id].monotonous is None:
                        neutral_regulators.append(regulator)
                    else:
                        if node.regulator_states[0].edges[regulator.id].monotonous < 0:
                            inhibitor_mask += 1 << regulator.id
                        else:
                            activator_mask += 1 << regulator.id

                prototype_lattice = RegulatorLattice(graph, context.allowed_values, activator_mask, inhibitor_mask)
                prototype_lattice.initialise_patterns()
                regulator_lattices = [prototype_lattice]
                for regulator in neutral_regulators:
                    new_regulator_lattices = []
                    for j in range(0, regulator.maximum + 1):
                        for lattice in regulator_lattices:
                            new_regulator_lattices.append(lattice.add_neutral_regulator(regulator, j))

                    regulator_lattices = list(new_regulator_lattices)

                for lattice in regulator_lattices:
                    for limit in activation_limits.limits:
                        if lattice.neutral_regulators.matches(limit):
                            lattice.limit_activation(limit)

                    for limit in inhibition_limits.limits:
                        if lattice.neutral_regulators.matches(limit):
                            lattice.limit_inhibition(limit)

                node_header = "\"{0}\"".format(node.name)

                inhibition_string = "{0} -> {1}".format(value + 1, value)
                activation_string = "{0} -> {1}".format(value, value + 1)

                for lattice in regulator_lattices:
                    if can_activate:
                        for pattern in lattice.activation_patterns:
                            data += "{0} {1} {2}\n".format(node_header, activation_string, lattice.constraints(pattern))
                    if can_inhibit:
                        for pattern in lattice.inhibition_patterns:
                            data += "{0} {1} {2}\n".format(node_header, inhibition_string, lattice.constraints(pattern))

        marking_string = ""
        for node in graph.nodes:
            marking_string += ", \"{0}\"={1}".format(node.name, marking[node.id])
        marking_string = marking_string[2:]

        data += "initial_context {0}\n".format(marking_string)

        super().__init__(data.encode(encoding='UTF-8'))

    def populate_popen_args(self, args, kwargs): #workaround until pypint is fixed
        kwargs["input"] = self.data
        super(pypint.InMemoryModel, self).populate_popen_args(args, kwargs)


class Transition:
    def __init__(self, regulator_state, regulator_mask):
        self.regulator_state = regulator_state
        self.regulator_mask = regulator_mask

    def __contains__(self, regulator_state):
        mask = self.regulator_mask & regulator_state.target.regulators
        for i in range(0, len(self.regulator_state)):
            if mask & (1 << i) and not regulator_state.matches_value(i, self.regulator_state[i]):
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

    for transition in model.local_transitions:
        target = graph.get_node(transition.a)

        if transition.i < transition.j:
            if activations[target.id] is None:
                activations[target.id] = 0

            activations[target.id] = max(activations[target.id], transition.j)
        else:
            if inhibitions[target.id] is None:
                inhibitions[target.id] = target.maximum

            inhibitions[target.id] = min(inhibitions[target.id], transition.j)

    for node in graph.nodes:
        if activations[node.id] is None:
            context.forbid_activation(node)
        else:
            context.soft_limit_max(node, activations[node.id])

        if inhibitions[node.id] is None:
            context.forbid_inhibition(node)
        else:
            context.soft_limit_min(node, inhibitions[node.id])

