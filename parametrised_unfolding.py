import numpy
import time

verbose = False
target = 0
report_frequency = 4096


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
    def __init__(self, graph, initial_marking, initial_context):
        self.conditions = []
        self.events = []
        self.discarded_events = []
        self.initial_marking = list(initial_marking)
        self.initial_context = initial_context.copy()
        self.marking_table = [0]
        self.graph = graph
        for n in graph.nodes:
            for i in range(0,((n.maximum // 2) + 1)):
                self.marking_table += self.marking_table

    def add_event(self, pe_queue, add_extensions = True):
        if len(pe_queue) <= 0:
            return

        event = pe_queue.pop()
        self.events.append(event)

        table_entry = self.get_table_entry(event.marking)
        event.cutoff |= (event.marking == self.initial_marking) or table_entry.is_cutoff(event)
        if not event.cutoff:
            backhandCutoffs = table_entry.add_context(event.parameter_context.bounds(), event)
            for bc in backhandCutoffs:
                self.remove_suffix(bc, pe_queue)

        if verbose:
            print('Adding ' + str(event))
            print('Event count: ' + str(len(self.events)))

        for c in event.preset:
            cond = c.copy()
            cond.id = len(self.conditions)
            if c.node == event.target:
                cond.value = event.target_value
            cond.parent = event

            parent_coset = 0
            for cc in event.preset:
                if not parent_coset:
                    parent_coset = set(cc.coset)
                else:
                    parent_coset &= cc.coset

            event.poset.add(cond)
            self.conditions.append(cond)

            if event.cutoff or event.goal:
                continue

            cond.coset |= parent_coset
            cond.coset |= event.poset
            cond.coset -= event.preset

            for cc in cond.coset:
                cc.coset.add(cond)

            if add_extensions:
                possible_extension(self, cond, pe_queue)

        return event

    def get_table_entry(self, marking):
        index = 0
        for n in self.graph.nodes:
            index *= (2 ** ((n.maximum // 2) + 1))
            index += marking[n.id]

        if not self.marking_table[index]:
            self.marking_table[index] = MarkingTableEntry()

        return self.marking_table[index]

    def remove_suffix(self, event, queue):
        for cond in event.poset:
            for e in cond.poset:
                self.remove_event(e, queue)
            cond.poset = set()

    def remove_condition(self, cond, queue):
        for event in cond.poset:
            self.remove_event(event, queue)

        for cocond in cond.coset:
            if cocond != cond:
                try:
                    cocond.coset.remove(cond)
                except KeyError:
                    pass

        try:
            self.conditions.remove(cond)
        except ValueError:
            pass

    def remove_event(self, event, queue):
        self.discarded_events.append(event)

        for cond in event.poset:
            self.remove_condition(cond, queue)

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
        self.node = 0
        self.value = 0
        self.parent = 0
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
        self.target = 0
        self.target_value = 0
        self.nature = 1
        self.context = 0
        self.preset = set()
        self.poset = set()
        self.marking = []
        self.local_configuration = set()
        self.local_configuration.add(self)
        self.parikh = 0
        self.foata = 0
        self.parameter_context = 0
        self.cutoff = False
        self.goal = False

    def init_from_preset(self, initial_marking, initial_context):
        self.marking = list(initial_marking)

        for c in self.preset:
            if c.parent:
                self.local_configuration |= c.parent.local_configuration
                if not self.parameter_context:
                    self.parameter_context = c.parent.parameter_context.copy()
                else:
                    self.parameter_context.intersect(c.parent.parameter_context)

        for e in self.local_configuration:
            self.marking[e.target.id] += e.nature

        if not self.parameter_context:
            self.parameter_context = initial_context.copy()

        if self.nature > 0:
            self.parameter_context.limit_min(self.context, self.target_value)
        else:
            self.parameter_context.limit_max(self.context, self.target_value)

    def compute_foata(self):
        self.foata = []
        temp_events = set(self.local_configuration)

        foata_level = set()
        for e in self.local_configuration:
            if len(e.local_configuration) == 1:
                foata_level.add(e)
                temp_events.remove(e)

        while len(foata_level):
            self.foata.append(compute_parkih_vector(foata_level))
            foata_level = set()
            for e in temp_events:
                this_level = True
                for ee in e.local_configuration:
                    if (ee in temp_events) and (ee != e):
                        this_level = False
                        break
                if this_level:
                    foata_level.add(e)

            temp_events -= foata_level

    def compare(self, event):
        if len(self.local_configuration) != len(event.local_configuration):
            return len(self.local_configuration) - len(event.local_configuration)

        if not self.parikh:
            self.parikh = compute_parkih_vector(self.local_configuration)

        if not event.parikh:
            event.parikh = compute_parkih_vector(event.local_configuration)

        res = parikh_compare(self.parikh, event.parikh)
        if res:
            return res

        if not self.foata:
            self.compute_foata()

        if not event.foata:
            event.compute_foata()

        return foata_compare(self.foata, event.foata)

    def __str__(self):
        prestr = ''
        for c in self.preset:
            prestr += (',' + str(c))
        prestr = prestr[1:]

        return '{' + prestr + '}->' + self.target.name + str(self.target_value)


class MarkingTableEntry:
    def __init__(self):
        self.events = set()
        self.contexts = []

    def add_event(self, event):
        self.events.add(event)

    def add_context(self, hc, event):
        dim = hc.dimension()
        found = False
        i = 0
        while i < len(self.contexts):
            cdim = self.contexts[i].dimension()
            if cdim > dim:
                if hc.issubset(self.contexts[i]):
                    found = True
                    break
            elif cdim == dim:
                dist = self.contexts[i].distance(hc)
                if not dist:
                    found = True
                    break
                elif dist == 1:
                    union = self.contexts[i].union(hc)
                    self.contexts.remove(self.contexts[i])
                    self.add_context(union)
                    found = True
                    break
            elif cdim < dim:
                if not found:
                    found = True
                    self.contexts.insert(i, hc)
                    i += 1
                if self.contexts[i].issubset(hc):
                    self.contexts.remove(self.contexts[i])
                    i -= 1
            i += 1

        if not found:
            self.contexts.append(hc)

        backwardsCutoffs = set()
        for e in self.events:
            if (not e.cutoff) and (e != event) and e.parameter_context.bounds().issubset(hc):
                e.cutoff = True
                backwardsCutoffs.add(e)
        return backwardsCutoffs

    def is_cutoff(self, event):
        for c in self.contexts:
            if event.parameter_context.bounds().issubset(c):
                return True

        return False

    def is_possible(self, event):
        if event.parameter_context.empty():
            if verbose:
                print(str(event) + ' not possible, empty parameter context')
            return False

        for e in self.events:
            if (e.target == event.target) and (e.context == event.context):
                different = False
                for c in e.preset:
                    if c not in event.preset:
                        different = True
                        break
                if not different:
                    if verbose:
                        print(str(event) + ' not possible, already exists')
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

    def copy(self):
        copy = Lattice()
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

    def intersect(self, lattice):
        intersection = Lattice()

        if self.empty() or lattice.empty():
            return intersection

        intersection.min = numpy.maximum(self.min, lattice.min)
        intersection.max = numpy.minimum(self.max, lattice.max)

        return intersection

    def union(self, lattice):
        union = Lattice()

        union.min = self.min & lattice.min
        union.max = self.max | lattice.max

        return union


class ParameterContext:
    def __init__(self, graph = None):
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

    def bounds(self):
        return self.lattice

    def copy(self):
        copy = ParameterContext()
        copy.graph = self.graph
        copy.lattice = self.lattice.copy()
        for node in self.open_infima:
            copy.open_infima[node] = set(self.open_infima[node])
        for node in self.open_suprema:
            copy.open_suprema[node] = set(self.open_suprema[node])
        copy.observable_edges = set(self.observable_edges)

        return copy

    def limit(self, regulator_state, value):
        if self.lattice.limit(regulator_state.id, value):
            self.check_edge_labels(regulator_state)

    def limit_min(self, regulator_state, value):
        if self.lattice.limit_min(regulator_state.id, value):
            self.check_edge_labels(regulator_state)

    def limit_max(self, regulator_state, value):
        if self.lattice.limit_max(regulator_state.id, value):
            self.check_edge_labels(regulator_state)

    def intersect(self, pc):
        changed_indices = (self.lattice.min ^ pc.lattice.min) | (self.lattice.max ^ pc.lattice.max)

        self.lattice = self.lattice.intersect(pc.lattice)

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


def compute_monotonicity_extremes(node, positive):
    if not node.regulatory_states:
        return set()

    inhibitors = []
    activators = []

    for edge in node.regulatory_states[0].edges:
        if not edge:
            continue
        if edge.monotonous < 0:
            inhibitors.append(edge.source)
        elif edge.monotonous > 0:
            activators.append(edge.source)

    extremes = set()

    for regulator_state in node.regulatory_states:
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
            if (not i) or (event.context.id > parikh[i - 1].context.id) or\
                    ((event.context.id == parikh[i-1].context.id) and (event.target_value >= parikh[i-1].target_value)):
                if (not parikh[i]) or (event.context.id < parikh[i].context.id) or\
                        ((event.context.id == parikh[i].context.id) and (event.target_value < parikh[i].target_value)):
                    parikh[i] = event

    return parikh


def parikh_compare(vec1, vec2):
    for i in range(0, min(len(vec1), len(vec2))):
        if vec1[i].context.id != vec2[i].context.id:
            return vec1[i].context.id - vec2[i].context.id
        elif vec1[i].target_value != vec2[i].target_value:
            return vec1[i].target_value - vec2[i].target_value

    return len(vec1) - len(vec2)


def foata_compare(foata1, foata2):
    for i in range(0, min(len(foata1), len(foata2))):
        result= parikh_compare(foata1[i], foata2[i])
        if result:
            return result

    return len(foata1) - len(foata2)


def possible_extension(unfolding, condition, queue):
    coset_nodes = 0
    node_cosets = [0] * len(unfolding.graph.nodes)
    for cocond in condition.coset:
        if not node_cosets[cocond.node.id]:
            node_cosets[cocond.node.id] = []
        node_cosets[cocond.node.id].append(cocond)
        if not (coset_nodes & (1 << cocond.node.id)):
            coset_nodes += (1 << cocond.node.id)

    for node in unfolding.graph.nodes:
        if ((node.regulators ^ coset_nodes) & node.regulators) or (not node_cosets[node.id]):
            continue

        prefab_presets = []
        if node.regulators & (1 << node.id):
            prefab_presets.append(set())
        else:
            for cocond in node_cosets[node.id]:
                prefab_preset = set()
                prefab_preset.add(cocond)
                prefab_presets.append(prefab_preset)

        if not prefab_presets:
            continue

        for regulator_state in node.regulatory_states:
            possible_presets = list(prefab_presets)
            for i in range(0,len(unfolding.graph.nodes)):
                if (node.regulators & (1 << i)) and node_cosets[i]:
                    new_possible_presets = []
                    for cocond in node_cosets[i]:
                        if regulator_state.edges[i].threshold:
                            if ((regulator_state.regulators[i] == 0) and
                                        (cocond.value < regulator_state.edges[i].threshold)) or\
                                    ((regulator_state.regulators[i] == unfolding.graph.nodes[i].maximum) and
                                         (cocond.value >= regulator_state.edges[i].threshold)):
                                for possible_preset in possible_presets:
                                    if (possible_preset.issubset(cocond.coset)):
                                        npp = set(possible_preset)
                                        npp.add(cocond)
                                        new_possible_presets.append(npp)
                        else:
                            if regulator_state.regulators[i] == cocond.value:
                                for possible_preset in possible_presets:
                                    if (possible_preset.issubset(cocond.coset)):
                                        npp = set(possible_preset)
                                        npp.add(cocond)
                                        new_possible_presets.append(npp)
                    if not new_possible_presets:
                        break
                    else:
                        possible_presets = new_possible_presets

            for possible_preset in possible_presets:
                preset_hash = 0
                for cocond in possible_preset:
                    preset_hash += (1 << cocond.node.id)

                if node.regulators & (1 << node.id):
                    if preset_hash != node.regulators:
                        continue
                else:
                    if preset_hash != (node.regulators + (1 << node.id)):
                        continue

                target_cond = 0
                for cocond in possible_preset:
                    if cocond.node == node:
                        target_cond = cocond
                        break

                if not target_cond:
                    continue

                if target_cond.value > 0:
                    inhibit_event = Event()
                    inhibit_event.target = node
                    inhibit_event.target_value = (target_cond.value - 1)
                    inhibit_event.nature = -1
                    inhibit_event.context = regulator_state
                    inhibit_event.preset = possible_preset
                    enqueue_event(unfolding, queue, inhibit_event)
                if target_cond.value < node.maximum:
                    activ_event = Event()
                    activ_event.target = node
                    activ_event.target_value = (target_cond.value + 1)
                    activ_event.nature = 1
                    activ_event.context = regulator_state
                    activ_event.preset = possible_preset
                    enqueue_event(unfolding, queue, activ_event)


def enqueue_event(unfolding, queue, event):
    event.init_from_preset(unfolding.initial_marking, unfolding.initial_context)
    table_entry = unfolding.get_table_entry(event.marking)
    if table_entry.is_possible(event):
        if target:
            event.goal = (event.marking == target)
        table_entry.add_event(event)

        event.id = len(unfolding.events) + len(unfolding.discarded_events) + len(queue)
        for c in event.preset:
            c.poset.add(event)

        queue.push(event)


def init_unfolding(graph, initial_marking = None, initial_context = None):
    if not initial_marking:
        initial_marking = []
        for n in graph.nodes:
            initial_marking.append(n.initial)

    if not initial_context:
        initial_context = ParameterContext(graph)

    unfolding = Unfolding(graph, initial_marking, initial_context)

    for n in graph.nodes:
        cond = Condition()
        cond.id = len(unfolding.conditions)
        cond.node = n
        cond.value = initial_marking[n.id]
        for c in unfolding.conditions:
            c.coset.add(cond)
            cond.coset.add(c)
        unfolding.conditions.append(cond)

    return unfolding


def unfold(graph):
    unfolding = init_unfolding(graph)

    pe_queue = PossibleExtensionQueue()

    possible_extension(unfolding, unfolding.conditions[0], pe_queue)

    event = unfolding.add_event(pe_queue)

    while event:
        if not (len(unfolding.events) % report_frequency):
            print('Event count: ' + str(len(unfolding.events)))
            print('Event queue: ' + str(len(pe_queue)))

        event = unfolding.add_event(pe_queue)

    return unfolding
