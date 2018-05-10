import numpy
import automata_network
import parametrised_unfolding


class SoftLimitMarkingTableEntry(parametrised_unfolding.MarkingTableEntry):
    def __init__(self):
        super().__init__()

    def add_context(self, context):
        self.context_collection.add(context)

    def is_possible(self, event):
        if event.parameter_context.disabled() or \
                (event.parameter_context.allowed_values.min[event.target.id] < 0 and event.nature < 0) or \
                (event.parameter_context.allowed_values.min[event.target.id] > event.target_value and event.nature < 0) or \
                (event.parameter_context.allowed_values.max[event.target.id] < event.target_value and 0 < event.nature):
            if parametrised_unfolding.verbose:
                print("{0} not possible, does not lead to goal.".format(event))
            return False

        return super().is_possible(event)


class SoftLimitMarkingTable(parametrised_unfolding.MarkingTable):
    def __init__(self, graph):
        super().__init__(graph)

    def initialise_empty_entry(self, index):
        self.entries[index] = SoftLimitMarkingTableEntry()


class ReducibleLattice(parametrised_unfolding.Lattice):
    def __init__(self):
        super().__init__()

    def empty(self):
        if (self.min < 0).all() and (self.max < 0).all():
            return True

        mask = (self.min >= 0) * (self.max >= 0)
        return (self.min * mask > self.max * mask).any()

    def _initialise_child(self):
        return ReducibleLattice()

    def issubset(self, lattice):
        if lattice.empty():
            return self.empty()
        elif self.empty():
            return True

        min_mask = self.min >= 0
        max_mask = self.max >= 0

        return (min_mask * self.min >= min_mask * lattice.min).all() and \
               (min_mask * lattice.min >= 0).all() and \
               (max_mask * self.max <= max_mask * lattice.max).all()

    def limit_min(self, id, value):
        if self.min[id] < 0:
            return False

        super().limit_min(id, value)

    def limit_max(self, id, value):
        if self.max[id] < 0:
            return False

        super().limit_max(id, value)

    def forbid_inhibition(self, id):
        self.min[id] = -1

    def forbid_activation(self, id):
        self.max[id] = -1


class SoftLimitParameterContext(parametrised_unfolding.ParameterContext):
    def __init__(self, graph=None):
        super().__init__(graph)

        if graph is not None:
            self.allowed_values = ReducibleLattice()
            self.allowed_values.min = numpy.array([0] * len(self.graph.nodes))

            maximums = [0] * len(self.graph.nodes)
            for node in graph.nodes:
                maximums[node.id] = node.maximum

            self.allowed_values.max = numpy.array(maximums)

    def disabled(self):
        return (not self.allowed_values) or self.allowed_values.empty()

    def copy(self):
        copy = SoftLimitParameterContext()
        super()._populate_copy(copy)
        copy.allowed_values = self.allowed_values.copy()

        return copy

    def issubset(self, context):
        return super().issubset(context) and self.allowed_values.issubset(context.allowed_values)

    def intersect(self, context):
        super().intersect(context)

        self.allowed_values = self.allowed_values.intersection(context.allowed_values)

    def soft_limit(self, node, value):
        self.allowed_values.limit(node.id, value)

    def soft_limit_min(self, node, value):
        self.allowed_values.limit_min(node.id, value)

    def soft_limit_max(self, node, value):
        self.allowed_values.limit_max(node.id, value)

    def forbid_inhibition(self, node):
        self.allowed_values.forbid_inhibition(node.id)

    def forbid_activation(self, node):
        self.allowed_values.forbid_activation(node.id)


class GoalDrivenUnfolder(parametrised_unfolding.Unfolder):
    def __init__(self, report_interval, graph, goal, initial_marking=None, initial_context=None):
        super().__init__(report_interval, graph, initial_marking, initial_context)

        self.goal = goal
        if self.goal.matches(self.prefix.initial_marking):
            self.prefix.initial_context.allowed_lattice.invalidate()
        else:
            self.reduce_for_goal(self.prefix.initial_context, self.prefix.initial_marking)

    def _build_initial_context(self):
        return SoftLimitParameterContext(self.graph)

    def _build_marking_table(self):
        return SoftLimitMarkingTable(self.graph)

    def _add_event(self, event):
        if event is None:
            return

        if self.goal.matches(event.marking):
            event.goal = True
        else:
            self.reduce_for_goal(event.parameter_context, event.marking)

        super()._add_event(event)

    def reduce_for_goal(self, context, marking):
        model = automata_network.ConfigurationWrapperModel(self.graph, context, marking)
        reduced = model.reduce_for_goal(str(self.goal), squeeze=False)
        automata_network.restrict_context_to_model(self.graph, context, reduced)
