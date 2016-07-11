import numpy
import time

verbose = False
target = 0
report_frequency = 4096

class Unfolding:
    def __init__(self, graph):
        self.conditions = []
        self.events = []
        self.initial_marking = ([0] * len(graph.nodes))
        self.marking_table = [0]
        for n in graph.nodes:
            for i in range(0,((n.maximum // 2) + 1)):
                self.marking_table += self.marking_table

    def get_table_entry(self, marking, graph):
        index = 0
        for n in graph.nodes:
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

    def init_from_preset(self, initial_marking, graph):
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
            self.parameter_context = ParameterContext(graph)

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

        #self.add_context(event.parameter_context.interval)

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
            if (not e.cutoff) and (e != event) and e.parameter_context.interval.issubset(hc):
                e.cutoff = True
                backwardsCutoffs.add(e)
        return backwardsCutoffs

    def is_cutoff(self, event):
        for c in self.contexts:
            if event.parameter_context.interval.issubset(c):
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


class Hypercube:
    def __init__(self):
        self.invalidate()

    def full_context(graph):
        fc = Hypercube()

        fc.min = numpy.array([0] * graph.parametrisation_size)
        fc.max = numpy.array([1] * graph.parametrisation_size)
        for i in graph.contexts:
            fc.max[i] = graph.contexts[i].target.maximum

        return fc

    def invalidate(self):
        self.min = numpy.array([1])
        self.max = numpy.array([0])

    def empty(self):
        return (self.min > self.max).any()

    def copy(self):
        copy = Hypercube()
        copy.min = numpy.array(self.min)
        copy.max = numpy.array(self.max)

        return copy

    def dimension(self):
        return numpy.sum(self.min ^ self.max)

    def distance(self, hc):
        return numpy.sum(self.min ^ hc.min) + numpy.sum(self.max ^ hc.max)

    def limit(self, context, value):
        res = False
        res |= self.limit_min(context, value)
        res |= self.limit_max(context, value)

        return res

    def limit_min(self, context, value):
        if (not self.empty()) and (value > self.min[context]):
            self.min[context] = value
            return True

        return False

    def limit_max(self, context, value):
        if (not self.empty()) and (value < self.max[context]):
            self.max[context] = value
            return True

        return False

    def intersect(self, hc):
        intersection = Hypercube()

        if self.empty() or hc.empty():
            return intersection

        intersection.min = numpy.maximum(self.min, hc.min)
        intersection.max = numpy.minimum(self.max, hc.max)

        return intersection

    def issubset(self, hc):
        if hc.empty():
            return self.empty()
        elif self.empty():
            return True

        return (self.min >= hc.min).all() and (self.max <= hc.max).all()

    def union(self, hc):
        union = Hypercube()

        union.min = self.min & hc.min
        union.max = self.max | hc.max

        return union


class ParameterContext:
    def __init__(self, graph = 0):
        self.open_contexts = dict()
        if graph:
            self.graph = graph
            self.interval = Hypercube.full_context(graph)
            for node in graph.nodes:
                self.open_contexts[node] = set(node.contexts)
            self.observable_edges = set()

            for kp in graph.known_parameters:
                self.limit(graph.contexts[kp],graph.known_parameters[kp])
            for km in graph.known_minimums:
                self.limit_min(graph.contexts[km],graph.known_minimums[km])
            for km in graph.known_maximums:
                self.limit_max(graph.contexts[km],graph.known_maximums[km])

    def empty(self):
        return (not self.interval) or self.interval.empty()

    def copy(self):
        copy = ParameterContext()
        copy.graph = self.graph
        copy.interval = self.interval.copy()
        for node in self.open_contexts:
            copy.open_contexts[node] = set(self.open_contexts[node])
        copy.observable_edges = set(self.observable_edges)

        return copy

    def limit(self, context, value):
        if self.interval.limit(context.id, value):
            self.check_edge_labels(context)

    def limit_min(self, context, value):
        if self.interval.limit_min(context.id, value):
            self.check_edge_labels(context)

    def limit_max(self, context, value):
        if self.interval.limit_max(context.id, value):
            self.check_edge_labels(context)

    def intersect(self, pc):
        changed_indices = (self.interval.min ^ pc.interval.min) | (self.interval.max ^ pc.interval.max)

        self.interval = self.interval.intersect(pc.interval)

        for i in range(0, len(changed_indices)):
            if changed_indices[i]:
                self.check_edge_labels(self.graph.contexts[i])

    def check_edge_labels(self, context):
        #t = time.clock()
        self.check_observable(context)
        for e in context.edges:
            if e:
                sub = context.subcontexts[e.source.id]
                super = context.supercontexts[e.source.id]
                if e.monotonous:
                    if e.monotonous > 0:
                        if sub:
                            self.enforce_plus_monotonicity(sub, context)
                        if super:
                            self.enforce_plus_monotonicity(context, super)
                    else:
                        if sub:
                            self.enforce_minus_monotonicity(sub, context)
                        if super:
                            self.enforce_minus_monotonicity(context, super)
        #t = time.clock() - t
        #print('{0:.6f}'.format(t))

    def enforce_plus_monotonicity(self, sub, super):
        self.enforce_monotonicity(sub, super)

    def enforce_minus_monotonicity(self, sub, super):
        self.enforce_monotonicity(super, sub)

    def enforce_monotonicity(self, context0, context1):
        if self.empty():
            return

        if self.interval.min[context0.id] > 0:
            self.limit_min(context1, self.interval.min[context0.id])
        elif self.interval.max[context1.id] < context1.target.maximum:
            self.limit_max(context0, self.interval.max[context1.id])

    def check_observable(self, context):
        if self.empty() or (context not in self.open_contexts[context.target]):
            return

        if self.interval.min[context.id] == self.interval.max[context.id]:
            self.open_contexts[context.target].remove(context)

        for edge in context.edges:
            if edge and edge.observable and (edge not in self.observable_edges):
                observable = False
                sub = context.subcontexts[edge.source.id]
                super = context.supercontexts[edge.source.id]

                while sub:
                    if (self.interval.min[context.id] > self.interval.max[sub.id]) or (self.interval.min[sub.id] > self.interval.max[context.id]):
                        self.observable_edges.add(edge)
                        observable |= True
                        break
                    sub = sub.subcontexts[edge.source.id]

                while super:
                    if (self.interval.min[context.id] > self.interval.max[super.id]) or (self.interval.min[super.id] > self.interval.max[context.id]):
                        self.observable_edges.add(edge)
                        observable |= True
                        break
                    super = super.supercontexts[edge.source.id]

                if not observable:
                    if len(self.open_contexts) == 1:
                        self.enforce_observability(edge)
                    elif len(self.open_contexts) == 0:
                        self.interval.invalidate()

    def enforce_observability(self, edge):
        for open_context in self.open_contexts[edge.target]:
            pair = open_context.subcontexts[edge.source.id]
            if not pair:
                pair = open_context.supercontexts[edge.source.id]

            if open_context.target().maximum == 1:
                self.limit(open_context, (1 - self.interval.min[pair.id]))
            elif self.interval.min[pair.id] == 0:
                self.limit_min(open_context, 1)
            elif self.interval.max[pair.id] == open_context.target().maximum:
                self.limit_max(open_context, (open_context.target().maximum - 1))


def compute_parkih_vector(events):
    parikh = [0] * len(events)

    for i in range(0,len(events)):
        for e in events:
            if (not i) or (e.context.id > parikh[i - 1].context.id) or ((e.context.id == parikh[i-1].context.id) and (e.target_value >= parikh[i-1].target_value)):
                if (not parikh[i]) or (e.context.id < parikh[i].context.id) or ((e.context.id == parikh[i].context.id) and (e.target_value < parikh[i].target_value)):
                    parikh[i] = e

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
        res = parikh_compare(foata1[i], foata2[i])
        if res:
            return res

    return len(foata1) - len(foata2)


def possible_extension(graph, unfolding, condition, queue):
    coset_nodes = 0
    node_cosets = [0] * len(graph.nodes)
    for cocond in condition.coset:
        if not node_cosets[cocond.node.id]:
            node_cosets[cocond.node.id] = []
        node_cosets[cocond.node.id].append(cocond)
        if not (coset_nodes & (1 << cocond.node.id)):
            coset_nodes += (1 << cocond.node.id)

    for node in graph.nodes:
        if ((node.regulators ^ coset_nodes) & node.regulators) or (not node_cosets[node.id]):
            continue

        prefab_presets = []
        if node.regulators & (1 << node.id):
            prefab_presets.append(set())
        else:
            for cocond in node_cosets[node.id]:
                pp = set()
                pp.add(cocond)
                prefab_presets.append(pp)

        if not prefab_presets:
            continue

        for context in node.contexts:
            possible_presets = list(prefab_presets)
            for i in range(0,len(graph.nodes)):
                if (node.regulators & (1 << i)) and node_cosets[i]:
                    new_possible_presets = []
                    for cocond in node_cosets[i]:
                        if context.edges[i].threshold:
                            if ((context.regulators[i] == 0) and (cocond.value < context.edges[i].threshold)) or ((context.regulators[i] == graph.nodes[i].maximum) and (cocond.value >= context.edges[i].threshold)):
                                for pp in possible_presets:
                                    if (pp.issubset(cocond.coset)):
                                        npp = set(pp)
                                        npp.add(cocond)
                                        new_possible_presets.append(npp)
                        else:
                            if context.regulators[i] == cocond.value:
                                for pp in possible_presets:
                                    if (pp.issubset(cocond.coset)):
                                        npp = set(pp)
                                        npp.add(cocond)
                                        new_possible_presets.append(npp)
                    if not new_possible_presets:
                        break
                    else:
                        possible_presets = new_possible_presets

            for pp in possible_presets:
                pp_num = 0
                for cocond in pp:
                    pp_num += (1 << cocond.node.id)

                if node.regulators & (1 << node.id):
                    if pp_num != node.regulators:
                        continue
                else:
                    if pp_num != (node.regulators + (1 << node.id)):
                        continue

                target_cond = 0
                for cocond in pp:
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
                    inhibit_event.context = context
                    inhibit_event.preset = pp
                    enqueue_event(graph, unfolding, queue, inhibit_event)
                if target_cond.value < node.maximum:
                    activ_event = Event()
                    activ_event.target = node
                    activ_event.target_value = (target_cond.value + 1)
                    activ_event.nature = 1
                    activ_event.context = context
                    activ_event.preset = pp
                    enqueue_event(graph, unfolding, queue, activ_event)


#time_counter = 0
#aggregate_time = 0
#loop_time = time.clock()
event_id = 0


def enqueue_event(graph, unfolding, queue, event):
    global event_id

    event.init_from_preset(unfolding.initial_marking, graph)
    table_entry = unfolding.get_table_entry(event.marking, graph)
    if table_entry.is_possible(event):
        if target:
            event.goal = (event.marking == target)
        table_entry.add_event(event)

        event.id = event_id
        event_id += 1
        for c in event.preset:
            c.poset.add(event)

        jump = len(queue)
        min_i = 0
        max_i = len(queue)
        index = 0
        while jump > 0:
            jump = (jump // 2) + (jump % 2)
            if (index == len(queue)) or (event.compare(queue[index]) < 0):
                if not index:
                    break
                max_i = index
                index = max(min_i, index - jump)
            else:
                if index == len(queue):
                    break
                min_i = index + 1
                index = min(max_i, index + jump)
            if min_i == max_i:
                index = min_i
                break
        queue.insert(index,event)


def unfold(graph):
    unfolding = Unfolding(graph)

    cond_id = 0

    for n in graph.nodes:
        cond = Condition()
        cond.id = cond_id
        cond_id += 1
        cond.node = n
        cond.value = n.initial
        unfolding.initial_marking[n.id] = n.initial
        for c in unfolding.conditions:
            c.coset.add(cond)
            cond.coset.add(c)
        unfolding.conditions.append(cond)

    pe_queue = []
    possible_extension(graph, unfolding, unfolding.conditions[0], pe_queue)

    while len(pe_queue) > 0:
        event = pe_queue[0]
        unfolding.events.append(event)
        pe_queue = pe_queue[1:]

        if not (len(unfolding.events) % report_frequency):
            print('Event count: ' + str(len(unfolding.events)))
            print('Event queue: ' + str(len(pe_queue)))

        table_entry = unfolding.get_table_entry(event.marking, graph)
        event.cutoff = (event.marking == unfolding.initial_marking) or table_entry.is_cutoff(event)
        if not event.cutoff:
            backhandCutoffs = table_entry.add_context(event.parameter_context.interval, event)
            for bc in backhandCutoffs:
                unfolding.remove_suffix(bc, pe_queue)

        if verbose:
            print('Adding ' + str(event))
            print('Event count: ' + str(len(unfolding.events)))

        for c in event.preset:
            cond = c.copy()
            cond.id = cond_id
            cond_id += 1
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
            unfolding.conditions.append(cond)

            if event.cutoff or event.goal:
                continue

            cond.coset |= parent_coset
            cond.coset |= event.poset
            cond.coset -= event.preset

            for cc in cond.coset:
                cc.coset.add(cond)

            possible_extension(graph, unfolding, cond, pe_queue)

    return unfolding
