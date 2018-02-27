import numpy
import automata_network
import parametrised_unfolding


class SoftLimitMarkingTableEntry(parametrised_unfolding.MarkingTableEntry):
    def __init__(self):
        super().__init__()

    def add_context(self, context):
        self.context_collection.add(context)

    def is_possible(self, event):
        if event.parameter_context.soft_empty() or event.parameter_context.allowed_lattice.min[event.regulator_state.id] > event.target_value or \
                event.parameter_context.allowed_lattice.max[event.regulator_state.id] < event.target_value:
            if parametrised_unfolding.verbose:
                print("{0} not possible, does not lead to goal.".format(event))
            return False

        return super().is_possible(event)


class SoftLimitParameterContext(parametrised_unfolding.ParameterContext):
    def __init__(self, graph=None):
        super().__init__(graph)

        if graph is not None:
            self.allowed_lattice = parametrised_unfolding.Lattice()
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

        self.allowed_lattice = self.allowed_lattice.intersect(context.allowed_lattice)

    def soft_limit(self, context, value):
        self.allowed_lattice.limit(context.id, value)

    def soft_limit_min(self, context, value):
        self.allowed_lattice.limit_min(context.id, value)

    def soft_limit_max(self, context, value):
        self.allowed_lattice.limit_max(context.id, value)


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
            try :
                reduced = model.reduce_for_goal(str(self.goal))
                automata_network.restrict_context_to_model(event, reduced)
            except Exception as e:
                print("Reduction failed for event id: {0}".format(event.id))
                print("Pint ended with {0}. Error output: {1}".format(e.returncode, e.stderr))
                print("Model data:")
                print(model.data)

        super()._add_event(event)