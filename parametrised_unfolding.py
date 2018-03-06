import numpy


verbose = False


class PossibleExtensionQueue:
    def __init__(self):
        self.possible_extensions = []

    def __len__(self):
        return len(self.possible_extensions)

    def pop(self):
        if not self.possible_extensions:
            return

        top = self.possible_extensions[0]
        self.possible_extensions = self.possible_extensions[1:]

        return top

    def push(self, event):
        jump = len(self)
        min_i = 0
        max_i = len(self)
        index = 0
        while jump > 0:
            jump = (jump // 2) + (jump % 2)
            if (index == len(self)) or (event.compare(self.possible_extensions[index]) < 0):
                if not index:
                    break
                max_i = index
                index = max(min_i, index - jump)
            else:
                if index == len(self):
                    break
                min_i = index + 1
                index = min(max_i, index + jump)
            if min_i == max_i:
                index = min_i
                break
        self.possible_extensions.insert(index, event)

    def remove(self, event):
        self.possible_extensions.remove(event)


class Unfolding:
    def __init__(self, initial_marking, initial_context):
        self.conditions = []
        self.events = []
        self.discarded_events = []
        self.initial_marking = list(initial_marking)
        self.initial_context = initial_context.copy()

    def add_event(self, event):
        self.events.append(event)

        if verbose:
            print('Adding ' + str(event))
            print('Event count: ' + str(len(self.events)))

        for condition in event.preset:
            new_condition = condition.copy()
            new_condition.id = len(self.conditions)
            if condition.node == event.target:
                new_condition.value = event.target_value
            new_condition.parent = event

            parent_coset = None
            for parent_condition in event.preset:
                if parent_coset is None:
                    parent_coset = set(parent_condition.coset)
                else:
                    parent_coset &= parent_condition.coset

            event.poset.add(new_condition)
            self.conditions.append(new_condition)

            if event.cutoff or event.goal:
                continue

            new_condition.coset |= parent_coset
            new_condition.coset |= event.poset
            new_condition.coset -= event.preset

            for concurrent_condition in new_condition.coset:
                concurrent_condition.coset.add(new_condition)

    def remove_suffix(self, event, queue):
        for condition in event.poset:
            for successor_event in condition.poset:
                self.remove_event(successor_event, queue)
            condition.poset = set()

    def remove_condition(self, condition, queue):
        for event in condition.poset:
            self.remove_event(event, queue)

        for concurrent_condition in condition.coset:
            if concurrent_condition != condition:
                try:
                    concurrent_condition.coset.remove(condition)
                except KeyError:
                    pass

        try:
            self.conditions.remove(condition)
        except ValueError:
            pass

    def remove_event(self, event, queue):
        self.discarded_events.append(event)

        for condition in event.poset:
            self.remove_condition(condition, queue)

        try:
            self.events.remove(event)
        except ValueError:
            try:
                queue.remove(event)
            except ValueError:
                pass


class Condition:
    def __init__(self):
        self.id = 0
        self.node = None
        self.value = 0
        self.parent = None
        self.poset = set()
        self.coset = set()
        self.coset.add(self)

    def copy(self):
        copy = Condition()
        copy.node = self.node
        copy.value = self.value
        return copy

    def __str__(self):
        return str(self.node) + str(self.value) + '(c' + str(self.id) + ')'


class Event:
    def __init__(self):
        self.id = 0
        self.target = None
        self.target_value = 0
        self.nature = 1
        self.regulator_state = None
        self.preset = set()
        self.poset = set()
        self.marking = []
        self.local_configuration = set()
        self.local_configuration.add(self)
        self.parikh = None
        self.foata = None
        self.parameter_context = None
        self.cutoff = False
        self.goal = False

    def init_from_preset(self, initial_marking, initial_context):
        self.marking = list(initial_marking)

        for condition in self.preset:
            if condition.parent:
                self.local_configuration |= condition.parent.local_configuration
                if not self.parameter_context:
                    self.parameter_context = condition.parent.parameter_context.copy()
                else:
                    self.parameter_context.intersect(condition.parent.parameter_context)

        for event in self.local_configuration:
            self.marking[event.target.id] += event.nature

        if not self.parameter_context:
            self.parameter_context = initial_context.copy()

        if self.nature > 0:
            self.parameter_context.limit_min(self.regulator_state, self.target_value)
        else:
            self.parameter_context.limit_max(self.regulator_state, self.target_value)

    def compute_foata(self):
        self.foata = []
        temp_events = set(self.local_configuration)

        foata_level = set()
        for event in self.local_configuration:
            if len(event.local_configuration) == 1:
                foata_level.add(event)
                temp_events.remove(event)

        while len(foata_level):
            self.foata.append(compute_parkih_vector(foata_level))
            foata_level = set()
            for temp_event in temp_events:
                this_level = True
                for predecessor_event in temp_event.local_configuration:
                    if (predecessor_event in temp_events) and (predecessor_event != temp_event):
                        this_level = False
                        break
                if this_level:
                    foata_level.add(temp_event)

            temp_events -= foata_level

    def compare(self, event):
        if len(self.local_configuration) != len(event.local_configuration):
            return len(self.local_configuration) - len(event.local_configuration)

        if self.parikh is None:
            self.parikh = compute_parkih_vector(self.local_configuration)

        if event.parikh is None:
            event.parikh = compute_parkih_vector(event.local_configuration)

        result = parikh_compare(self.parikh, event.parikh)
        if result:
            return result

        if self.foata is None:
            self.compute_foata()

        if event.foata is None:
            event.compute_foata()

        return foata_compare(self.foata, event.foata)

    def __str__(self):
        preset_string = ''
        for condition in self.preset:
            preset_string += (',' + str(condition))
            preset_string = preset_string[1:]

        return '{' + preset_string + '}->' + self.target.name + str(self.target_value)


class ParameterContextCollection():
    def __init__(self):
        self.contexts = []

    def __len__(self):
        return len(self.contexts)

    def __contains__(self, context):
        for existing_context in self.contexts:
            if context.issubset(existing_context):
                return True

        return False

    def add(self, context):
        for i in range(0, len(self.contexts)):
            if context.issubset(self.contexts[i]):
                return

            if self.contexts[i].issubset(context):
                self.contexts.remove(self.contexts[i])
                i -= 1

        self.contexts.append(context)

    def insert(self, index, context):
        self.contexts.insert(index, context)

    def remove(self, context):
        self.contexts.remove(context)


class MarkingTableEntry:
    def __init__(self):
        self.events = set()
        self.context_collection = ParameterContextCollection()

    def add_event(self, event):
        self.events.add(event)

    def add_context(self, context):
        dimension = context.lattice.dimension()
        found = False
        i = 0
        while i < len(self.context_collection):
            existing_dimension = self.context_collection.contexts[i].lattice.dimension()
            if existing_dimension > dimension:
                if context.issubset(self.context_collection.contexts[i]):
                    found = True
                    break
            elif existing_dimension == dimension:
                distance = self.context_collection.contexts[i].lattice.distance(context.lattice)
                if not distance:
                    found = True
                    break
                elif distance == 1:
                    union = self.context_collection.contexts[i].union(context)
                    self.context_collection.remove(self.context_collection.contexts[i])
                    self.add_context(union)
                    found = True
                    break
            elif existing_dimension < dimension:
                if not found:
                    found = True
                    self.context_collection.insert(i, context)
                    i += 1
                if self.context_collection.contexts[i].issubset(context):
                    self.context_collection.remove(self.context_collection.contexts[i])
                    i -= 1
            i += 1

        if not found:
            self.context_collection.add(context)

    def obtain_backwards_cutoffs(self, event):
        backwards_cutoffs = set()
        for existing_event in self.events:
            if (not existing_event.cutoff) and (existing_event != event) and \
                    existing_event.parameter_context.issubset(event.parameter_context):
                existing_event.cutoff = True
                backwards_cutoffs.add(existing_event)
        return backwards_cutoffs

    def is_cutoff(self, event):
        return event.parameter_context in self.context_collection

    def is_possible(self, event):
        if event.parameter_context.empty():
            if verbose:
                print("{0} not possible, empty parameter context.".format(event))
            return False

        for existing_event in self.events:
            if (existing_event.target == event.target) and (existing_event.regulator_state == event.regulator_state):
                different = False
                for condition in existing_event.preset:
                    if condition not in event.preset:
                        different = True
                        break
                if not different:
                    if verbose:
                        print("{0} not possible, already exists".format(event))
                    return False
        return True


class Lattice:
    def __init__(self):
        self.invalidate()

    @staticmethod
    def full_context(graph):
        lattice = Lattice()

        lattice.min = numpy.array([0] * graph.parametrisation_size)
        lattice.max = numpy.array([1] * graph.parametrisation_size)
        for i in graph.regulator_states:
            lattice.max[i] = graph.regulator_states[i].target.maximum

        return lattice

    def empty(self):
        return (self.min > self.max).any()

    def size(self):
        return len(self.min)

    def dimension(self):
        return numpy.sum(self.max - self.min)

    def distance(self, lattice):
        if self.size() != lattice.size():
            return

        return numpy.sum(abs(self.min - lattice.min)) + numpy.sum(abs(self.max - lattice.max))

    def issubset(self, lattice):
        if lattice.empty():
            return self.empty()
        elif self.empty():
            return True

        return (self.min >= lattice.min).all() and (self.max <= lattice.max).all()

    # noinspection PyMethodMayBeStatic
    def _initialise_child(self):
        return Lattice()

    def copy(self):
        copy = self._initialise_child()
        copy.min = numpy.array(self.min)
        copy.max = numpy.array(self.max)

        return copy

    # noinspection PyAttributeOutsideInit
    def invalidate(self):
        self.min = numpy.array([1])
        self.max = numpy.array([0])

    def limit(self, regulator_state_id, value):
        res = False
        res |= self.limit_min(regulator_state_id, value)
        res |= self.limit_max(regulator_state_id, value)

        return res

    def limit_min(self, regulator_state_id, value):
        if (not self.empty()) and (value > self.min[regulator_state_id]):
            self.min[regulator_state_id] = value
            return True

        return False

    def limit_max(self, regulator_state_id, value):
        if (not self.empty()) and (value < self.max[regulator_state_id]):
            self.max[regulator_state_id] = value
            return True

        return False

    def intersection(self, lattice):
        intersection = self._initialise_child()

        if self.empty() or lattice.empty():
            return intersection

        intersection.min = numpy.maximum(self.min, lattice.min)
        intersection.max = numpy.minimum(self.max, lattice.max)

        return intersection

    def union(self, lattice):
        union = self._initialise_child()

        union.min = self.min & lattice.min
        union.max = self.max | lattice.max

        return union


class ParameterContext:
    def __init__(self, graph=None):
        self.open_suprema = dict()
        self.open_infima = dict()

        if graph is not None:
            self.graph = graph
            self.lattice = Lattice.full_context(graph)
            for node in graph.nodes:
                self.open_infima[node] = compute_monotonicity_extremes(node, False)
                self.open_suprema[node] = compute_monotonicity_extremes(node, True)
            self.observable_edges = set()

            for kp in graph.known_parameters:
                self.limit(graph.regulator_states[kp], graph.known_parameters[kp])
            for km in graph.known_minimums:
                self.limit_min(graph.regulator_states[km], graph.known_minimums[km])
            for km in graph.known_maximums:
                self.limit_max(graph.regulator_states[km], graph.known_maximums[km])

    def empty(self):
        return (not self.lattice) or self.lattice.empty()

    def copy(self):
        copy = ParameterContext()
        self._populate_copy(copy)

        return copy

    def _populate_copy(self, copy):
        copy.graph = self.graph
        copy.lattice = self.lattice.copy()
        for node in self.open_infima:
            copy.open_infima[node] = set(self.open_infima[node])
        for node in self.open_suprema:
            copy.open_suprema[node] = set(self.open_suprema[node])
        copy.observable_edges = set(self.observable_edges)

    def issubset(self, context):
        return self.lattice.issubset(context.lattice)

    def limit(self, regulator_state, value):
        if self.lattice.limit(regulator_state.id, value):
            self.check_edge_labels(regulator_state)

    def limit_min(self, regulator_state, value):
        if self.lattice.limit_min(regulator_state.id, value):
            self.check_edge_labels(regulator_state)

    def limit_max(self, regulator_state, value):
        if self.lattice.limit_max(regulator_state.id, value):
            self.check_edge_labels(regulator_state)

    def intersect(self, context):
        changed_indices = (self.lattice.min ^ context.lattice.min) | (self.lattice.max ^ context.lattice.max)

        self.lattice = self.lattice.intersection(context.lattice)

        for i in range(0, len(changed_indices)):
            if changed_indices[i]:
                self.check_edge_labels(self.graph.regulator_states[i])

    def union(self, context):
        union = self.copy()
        union.lattice = self.lattice.union(context.lattice)

        union.open_infima = dict()
        union.open_suprema = dict()

        for node in self.graph.nodes:
            union.open_infima[node] = compute_monotonicity_extremes(node, False)
            if union.lattice.min[union.open_infima[node].id] == union.lattice.max[union.open_infima[node].id]:
                union.close_infimum(union.open_infima[node])
            union.open_suprema[node] = compute_monotonicity_extremes(node, True)
            if union.lattice.min[union.open_suprema[node].id] == union.lattice.max[union.open_suprema[node].id]:
                union.close_supremum(union.open_suprema[node])

    def check_edge_labels(self, regulator_state):
        self.check_observable(regulator_state)
        for edge in regulator_state.edges:
            if edge:
                substate = regulator_state.substates[edge.source.id]
                superstate = regulator_state.superstates[edge.source.id]
                if edge.monotonous:
                    if edge.monotonous > 0:
                        if substate:
                            self.enforce_plus_monotonicity(substate, regulator_state)
                        if superstate:
                            self.enforce_plus_monotonicity(regulator_state, superstate)
                    else:
                        if substate:
                            self.enforce_minus_monotonicity(substate, regulator_state)
                        if superstate:
                            self.enforce_minus_monotonicity(regulator_state, superstate)

    def enforce_plus_monotonicity(self, substate, superstate):
        self.enforce_monotonicity(substate, superstate)

    def enforce_minus_monotonicity(self, substate, superstate):
        self.enforce_monotonicity(superstate, substate)

    def enforce_monotonicity(self, lesser_regulator_state, greater_regulator_state):
        if self.empty():
            return

        if self.lattice.min[lesser_regulator_state.id] > 0:
            self.limit_min(greater_regulator_state, self.lattice.min[lesser_regulator_state.id])
        if self.lattice.max[greater_regulator_state.id] < greater_regulator_state.target.maximum:
            self.limit_max(lesser_regulator_state, self.lattice.max[greater_regulator_state.id])

    def close_infimum(self, regulator_state):
        self.open_infima[regulator_state.target].remove(regulator_state)

        for edge in regulator_state.edges:
            if not edge:
                continue
            prime_filter = None
            if edge.monotonous < 0:
                prime_filter = regulator_state.substates[edge.source.id]
            if edge.monotonous > 0:
                prime_filter = regulator_state.superstates[edge.source.id]
            if prime_filter:
                self.open_infima[regulator_state.target].add(prime_filter)
                if self.lattice.min[prime_filter.id] == self.lattice.max[prime_filter.id]:
                    self.close_infimum(prime_filter)

    def close_supremum(self, regulator_state):
        self.open_suprema[regulator_state.target].remove(regulator_state)

        for edge in regulator_state.edges:
            if not edge:
                continue
            prime_ideal = None
            if edge.monotonous < 0:
                prime_ideal = regulator_state.superstates[edge.source.id]
            if edge.monotonous > 0:
                prime_ideal = regulator_state.substates[edge.source.id]
            if prime_ideal:
                self.open_suprema[regulator_state.target].add(prime_ideal)
                if self.lattice.min[prime_ideal.id] == self.lattice.max[prime_ideal.id]:
                    self.close_supremum(prime_ideal)

    def check_observable(self, regulator_state):
        if self.empty():
            return

        if self.lattice.min[regulator_state.id] == self.lattice.max[regulator_state.id]:
            if regulator_state in self.open_infima[regulator_state.target]:
                self.close_infimum(regulator_state)
            if regulator_state in self.open_suprema[regulator_state.target]:
                self.close_supremum(regulator_state)

        for edge in regulator_state.edges:
            if edge and edge.observable and (edge not in self.observable_edges):
                observable = False
                substate = regulator_state.substates[edge.source.id]
                superstate = regulator_state.superstates[edge.source.id]

                while substate:
                    if (self.lattice.min[regulator_state.id] > self.lattice.max[substate.id]) or\
                            (self.lattice.min[substate.id] > self.lattice.max[regulator_state.id]):
                        self.observable_edges.add(edge)
                        observable |= True
                        break
                    substate = substate.substates[edge.source.id]

                while superstate:
                    if (self.lattice.min[regulator_state.id] > self.lattice.max[superstate.id]) or\
                            (self.lattice.min[superstate.id] > self.lattice.max[regulator_state.id]):
                        self.observable_edges.add(edge)
                        observable |= True
                        break
                    superstate = superstate.superstates[edge.source.id]

                if not observable:
                    if len(self.open_infima) == 1:
                        self.enforce_observability_upper(edge)
                    if len(self.open_suprema) == 1:
                        self.enforce_observability_lower(edge)
                    if len(self.open_infima) + len(self.open_suprema) == 0:
                        self.lattice.invalidate()

    def enforce_observability_upper(self, edge):
        for infimum in self.open_infima[edge.target]:
            suprema_agree = True
            substate = infimum.substates[edge.source]
            superstate = infimum.superstates[edge.source]

            while substate:
                if self.lattice.max[infimum.id] != self.lattice.max[substate.id]:
                    suprema_agree = False
                    break
                substate = substate.substates[edge.source]

            while superstate:
                if self.lattice.max[infimum.id] != self.lattice.max[superstate.id]:
                    suprema_agree = False
                    break
                superstate = superstate.superstates[edge.source]

            if suprema_agree:
                self.limit_max(infimum, self.lattice.max[infimum.id] - 1)

    def enforce_observability_lower(self, edge):
        for supremum in self.open_suprema[edge.target]:
            infima_agree = True
            substate = supremum.substates[edge.source]
            superstate = supremum.superstates[edge.source]

            while substate:
                if self.lattice.min[supremum.id] != self.lattice.min[substate.id]:
                    infima_agree = False
                    break
                substate = substate.substates[edge.source]

            while superstate:
                if self.lattice.min[supremum.id] != self.lattice.min[superstate.id]:
                    infima_agree = False
                    break
                superstate = superstate.superstates[edge.source]

            if infima_agree:
                self.limit_min(supremum, self.lattice.min[supremum.id] + 1)


class Unfolder():
    def __init__(self, report_interval, graph, initial_marking=None, initial_context=None):
        self.graph = graph
        self.report_interval = report_interval

        if initial_marking is None:
            initial_marking = self._build_initial_marking()

        if initial_context is None:
            initial_context = self._build_initial_context()

        self.prefix = Unfolding(initial_marking, initial_context)

        for node in self.graph.nodes:
            initial_condition = Condition()
            initial_condition.id = len(self.prefix.conditions)
            initial_condition.node = node
            initial_condition.value = initial_marking[node.id]
            for condition in self.prefix.conditions:
                condition.coset.add(initial_condition)
                initial_condition.coset.add(condition)
            self.prefix.conditions.append(initial_condition)

        self.possible_extensions = PossibleExtensionQueue()
        self.marking_table = [None]

        for node in self.graph.nodes:
            for i in range(0,((node.maximum // 2) + 1)):
                self.marking_table += self.marking_table

    def _build_initial_marking(self):
        initial_marking = []
        for node in self.graph.nodes:
            initial_marking.append(node.initial)

        return initial_marking

    def _build_initial_context(self):
        return ParameterContext(self.graph)

    def _create_marking_table_entry(self, index):
        self.marking_table[index] = MarkingTableEntry()

    def _get_marking_table_entry(self, marking):
        index = 0
        for node in self.graph.nodes:
            index *= (2 ** ((node.maximum // 2) + 1))
            index += marking[node.id]

        if not self.marking_table[index]:
            self._create_marking_table_entry(index)

        return self.marking_table[index]

    def unfold(self):
        self._compute_possible_extensions(self.prefix.conditions[0])

        event = self.possible_extensions.pop()

        while event:
            self._add_event(event)
            if not (len(self.prefix.events) % self.report_interval):
                print('Event count: ' + str(len(self.prefix.events)))
                print('Event queue: ' + str(len(self.possible_extensions)))

            event = self.possible_extensions.pop()

    @staticmethod
    def _compute_condition_cosets_for_nodes(condition, node_cosets):
        coset_nodes = 0
        for concurrent_condition in condition.coset:
            if node_cosets[concurrent_condition.node.id] is None:
                node_cosets[concurrent_condition.node.id] = []
            node_cosets[concurrent_condition.node.id].append(concurrent_condition)
            if not (coset_nodes & (1 << concurrent_condition.node.id)):
                coset_nodes += (1 << concurrent_condition.node.id)

        return coset_nodes

    @staticmethod
    def _compute_prefab_extension_presets(node, node_cosets):
        prefab_presets = []
        if node.regulators & (1 << node.id):
            prefab_presets.append(set())
        else:
            for concurrent_condition in node_cosets[node.id]:
                prefab_preset = set()
                prefab_preset.add(concurrent_condition)
                prefab_presets.append(prefab_preset)

        return prefab_presets

    def _compute_possible_extension_presets(self, regulator_state, node, node_cosets, prefab_presets):
        possible_presets = list(prefab_presets)

        for i in range(0, len(self.graph.nodes)):
            if (node.regulators & (1 << i)) and node_cosets[i]:
                new_possible_presets = []

                for concurrent_condition in node_cosets[i]:
                    if regulator_state.edges[i].threshold:
                        if ((regulator_state.regulators[i] == 0) and
                            (concurrent_condition.value < regulator_state.edges[i].threshold)) or \
                                ((regulator_state.regulators[i] == self.graph.nodes[i].maximum) and
                                 (concurrent_condition.value >= regulator_state.edges[i].threshold)):
                            for possible_preset in possible_presets:
                                if possible_preset.issubset(concurrent_condition.coset):
                                    npp = set(possible_preset)
                                    npp.add(concurrent_condition)
                                    new_possible_presets.append(npp)
                    else:
                        if regulator_state.regulators[i] == concurrent_condition.value:
                            for possible_preset in possible_presets:
                                if possible_preset.issubset(concurrent_condition.coset):
                                    npp = set(possible_preset)
                                    npp.add(concurrent_condition)
                                    new_possible_presets.append(npp)

                if not new_possible_presets:
                    break
                else:
                    possible_presets = new_possible_presets

        return possible_presets

    @staticmethod
    def _validate_extension_preset(node, possible_preset):
        preset_hash = 0
        for concurrent_condition in possible_preset:
            preset_hash += (1 << concurrent_condition.node.id)

        regulator_hash = node.regulators
        if not node.regulators & (1 << node.id):
            regulator_hash += (1 << node.id)

        return preset_hash == regulator_hash

    @staticmethod
    def _get_target_condition(node, possible_preset):
        for concurrent_condition in possible_preset:
            if concurrent_condition.node == node:
                return concurrent_condition

        return None

    def _compute_possible_extensions_for_node(self, node, node_cosets):
        prefab_presets = Unfolder._compute_prefab_extension_presets(node, node_cosets)
        if not prefab_presets:
            return

        for regulator_state in node.regulator_states:
            possible_presets = self._compute_possible_extension_presets(regulator_state, node, node_cosets, prefab_presets)

            for possible_preset in possible_presets:
                if not Unfolder._validate_extension_preset(node, possible_preset):
                    continue

                target_condition = Unfolder._get_target_condition(node, possible_preset)
                if target_condition is None:
                    continue

                if target_condition.value > 0:
                    inhibition_event = Event()
                    inhibition_event.target = node
                    inhibition_event.target_value = (target_condition.value - 1)
                    inhibition_event.nature = -1
                    inhibition_event.regulator_state = regulator_state
                    inhibition_event.preset = possible_preset
                    self._enqueue_event(inhibition_event)
                if target_condition.value < node.maximum:
                    activation_event = Event()
                    activation_event.target = node
                    activation_event.target_value = (target_condition.value + 1)
                    activation_event.nature = 1
                    activation_event.regulator_state = regulator_state
                    activation_event.preset = possible_preset
                    self._enqueue_event(activation_event)

    def _compute_possible_extensions(self, condition):
        node_cosets = [None] * len(self.graph.nodes)
        coset_nodes = Unfolder._compute_condition_cosets_for_nodes(condition, node_cosets)

        for node in self.graph.nodes:
            if ((node.regulators ^ coset_nodes) & node.regulators) or (not node_cosets[node.id]):
                continue

            self._compute_possible_extensions_for_node(node, node_cosets)

    def _enqueue_event(self, event):
        event.init_from_preset(self.prefix.initial_marking, self.prefix.initial_context)

        table_entry = self._get_marking_table_entry(event.marking)
        if table_entry.is_possible(event):
            table_entry.add_event(event)

            event.id = len(self.prefix.events) + len(self.prefix.discarded_events) + len(self.possible_extensions)
            for condition in event.preset:
                condition.poset.add(event)

            self.possible_extensions.push(event)

    def _backwards_cutoff(self, event):
        self.prefix.remove_suffix(event, self.possible_extensions)

    def _add_event(self, event):
        if event is None:
            return

        table_entry = self._get_marking_table_entry(event.marking)
        event.cutoff |= (event.marking == self.prefix.initial_marking) or table_entry.is_cutoff(event)

        self.prefix.add_event(event)

        if not event.cutoff and not event.goal:
            table_entry.add_context(event.parameter_context)
            backwards_cutoffs = table_entry.obtain_backwards_cutoffs(event)
            for backwards_cutoff in backwards_cutoffs:
                self._backwards_cutoff(backwards_cutoff)

            for condition in event.poset:
                self._compute_possible_extensions(condition)


def compute_monotonicity_extremes(node, positive):
    if not node.regulator_states:
        return set()

    inhibitors = []
    activators = []

    for edge in node.regulator_states[0].edges:
        if not edge:
            continue
        if edge.monotonous < 0:
            inhibitors.append(edge.source)
        elif edge.monotonous > 0:
            activators.append(edge.source)

    extremes = set()

    for regulator_state in node.regulator_states:
        extreme = True
        for a in activators:
            if (not positive and regulator_state.substates[a.id]) or (positive and regulator_state.superstates[a.id]):
                extreme = False
                break

        for i in inhibitors:
            if (not positive and regulator_state.superstates[i.id]) or (positive and regulator_state.substates[i.id]):
                extreme = False
                break

        if extreme:
            extremes.add(regulator_state)

    return extremes


def compute_parkih_vector(events):
    parikh = [0] * len(events)

    for i in range(0,len(events)):
        for event in events:
            if (not i) or (event.regulator_state.id > parikh[i - 1].regulator_state.id) or\
                    ((event.regulator_state.id == parikh[i - 1].regulator_state.id) and (event.target_value >= parikh[i - 1].target_value)):
                if (not parikh[i]) or (event.regulator_state.id < parikh[i].regulator_state.id) or\
                        ((event.regulator_state.id == parikh[i].regulator_state.id) and (event.target_value < parikh[i].target_value)):
                    parikh[i] = event

    return parikh


def parikh_compare(vector1, vector2):
    for i in range(0, min(len(vector1), len(vector2))):
        if vector1[i].regulator_state.id != vector2[i].regulator_state.id:
            return vector1[i].regulator_state.id - vector2[i].regulator_state.id
        elif vector1[i].target_value != vector2[i].target_value:
            return vector1[i].target_value - vector2[i].target_value

    return len(vector1) - len(vector2)


def foata_compare(foata1, foata2):
    for i in range(0, min(len(foata1), len(foata2))):
        result= parikh_compare(foata1[i], foata2[i])
        if result:
            return result

    return len(foata1) - len(foata2)
