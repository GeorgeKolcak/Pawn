import numpy
import automata_network
import parametrised_unfolding


class SoftLimitMarkingTableEntry(parametrised_unfolding.MarkingTableEntry):
    def __init__(self):
        super().__init__()

    def add_context(self, context):
        self.context_collection.add(context)

    def is_possible(self, event):
        if event.parameter_context.soft_empty() or \
                (event.parameter_context.allowed_lattice.min[event.regulator_state.id] < 0 and event.nature < 0) or \
                (event.parameter_context.allowed_lattice.min[event.regulator_state.id] > event.target_value and event.nature < 0) or \
                (event.parameter_context.allowed_lattice.max[event.regulator_state.id] < event.target_value and 0 < event.nature):
            if parametrised_unfolding.verbose:
                print("{0} not possible, does not lead to goal.".format(event))
            return False

        return super().is_possible(event)


class ReducibleLattice(parametrised_unfolding.Lattice):
    def __init__(self):
        super().__init__()

    def empty(self):
        mask = numpy.sign(self.min) - numpy.sign(self.max) == 0
        return (self.min * mask > self.max * mask).any()

    def _initialise_child(self):
        return ReducibleLattice()

    def limit_min(self, regulator_state_id, value):
        if self.min[regulator_state_id] < 0:
            return False

        super().limit_min(regulator_state_id, value)

    def limit_max(self, regulator_state_id, value):
        if self.max[regulator_state_id] < 0:
            return False

        super().limit_max(regulator_state_id, value)

    def forbid_inhibition(self, regulator_state_id):
        self.min[regulator_state_id] = -1

    def forbid_activation(self, regulator_state_id):
        self.max[regulator_state_id] = -1


class SoftLimitParameterContext(parametrised_unfolding.ParameterContext):
    def __init__(self, graph=None):
        super().__init__(graph)

        if graph is not None:
            self.allowed_lattice = ReducibleLattice()
            self.allowed_lattice.min = numpy.array([0] * len(self.graph.regulator_states))
            self.allowed_lattice.max = numpy.array([1] * len(self.graph.regulator_states))

    def soft_empty(self):
        return (not self.allowed_lattice) or self.allowed_lattice.empty()

    def copy(self):
        copy = SoftLimitParameterContext()
        super()._populate_copy(copy)
        copy.allowed_lattice = self.allowed_lattice.copy()

        return copy

    def issubset(self, context):
        return super().issubset(context) and self.allowed_lattice.issubset(context.allowed_lattice)

    def intersect(self, context):
        super().intersect(context)

        self.allowed_lattice = self.allowed_lattice.intersection(context.allowed_lattice)

    def soft_limit(self, regulator_state, value):
        self.allowed_lattice.limit(regulator_state.id, value)

    def soft_limit_min(self, regulator_state, value):
        self.allowed_lattice.limit_min(regulator_state.id, value)

    def soft_limit_max(self, regulator_state, value):
        self.allowed_lattice.limit_max(regulator_state.id, value)

    def forbid_inhibition(self, regulator_state):
        self.allowed_lattice.forbid_inhibition(regulator_state.id)

    def forbid_activation(self, regulator_state):
        self.allowed_lattice.forbid_activation(regulator_state.id)


class GoalDrivenUnfolder(parametrised_unfolding.Unfolder):
    def __init__(self, report_interval, graph, goal, initial_marking=None, initial_context=None):
        super().__init__(report_interval, graph, initial_marking, initial_context)

        self.goal = goal

    def _build_initial_context(self):
        return SoftLimitParameterContext(self.graph)

    def _create_marking_table_entry(self, index):
        self.marking_table[index] = SoftLimitMarkingTableEntry()

    def _add_event(self, event):
        if event is None:
            return

        if self.goal.matches(event.marking):
            event.goal = True
        else:
            model = automata_network.ConfigurationWrapperModel(self.graph, event)
            reduced = model.reduce_for_goal(str(self.goal), squeeze=False)
            automata_network.restrict_context_to_model(event, reduced)

        super()._add_event(event)