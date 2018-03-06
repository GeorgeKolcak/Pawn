import sys
import time
import regulatory_network
import parametrised_unfolding
import goal_driven_unfolding


input_file_name = ""
output_file_name = ""

goal = None

timed = False
exec_time = 0
report_interval = 4096

args = []
i = 1
while i < len(sys.argv):
    if sys.argv[i][0] == '-':
        modifiers = sys.argv[i]
        for c in modifiers:
            if c == 'v':
                parametrised_unfolding.verbose = True
            elif c == 'm':
                regulatory_network.minmax = True
            elif c == 't':
                timed = True
            elif c == 'g':
                if i == (len(sys.argv) - 1):
                    print("The -g modifier is expected to be followed by a marking.")
                    exit(2)
                goal = sys.argv[i + 1]
                i += 1
            elif c == 'r':
                if i == (len(sys.argv) - 1):
                    print("The -r modifier is expected to be followed by a number.")
                    exit(2)
                report_interval = int(sys.argv[i + 1])
                i += 1
    else:
        args.append(sys.argv[i])
    i += 1

if len(args) < 1:
    print("No input file specified. Usage: Pawn <input file> [<output file>,-v,-g <target node>]")
    exit(1)
elif len(args) == 1:
    input_file_name = args[0]
    output_file_name = input_file_name.split(".")[0] + ".dot"
else:
    input_file_name = args[0]
    output_file_name = args[1]

if timed:
    exec_time = time.clock()

graph = regulatory_network.parse_regulatory_graph(input_file_name)

if goal is not None:
    goal_marking = regulatory_network.PartialState(graph)

    for specification in goal.split(','):
        values = specification.split('=')

        node = graph.get_node(values[0].strip())
        if node is None:
            print("The specified goal \"{0}\" does not match the system definition. Unknown component '{1}'.".format(goal, values[0].strip()))
            exit(2)

        goal_marking.mask[node.id] = True
        goal_marking.values[node.id] = int(values[1])

    unfolder = goal_driven_unfolding.GoalDrivenUnfolder(report_interval, graph, goal_marking)
else:
    unfolder = parametrised_unfolding.Unfolder(report_interval, graph)

unfolder.unfold()
unfolding = unfolder.prefix

output = open(output_file_name, 'w')

output.write("digraph {\n")

for condition in unfolding.conditions:
    output.write('c' + str(condition.id) + ' [label="' + condition.node.name + str(condition.value) + '(c' + str(condition.id) + ')" shape=circle];\n')

event_count = 0
cutoff_count = 0
for event in unfolding.events:
    event_count += 1
    context_string = ''
    for i in range(0, len(graph.nodes)):
        if event.regulator_state.regulators[i] > 0:
            context_string += (',' + graph.nodes[i].name + str(event.regulator_state.regulators[i]))
    context_string = context_string[1:]
    bounds_string = ''
    if not event.parameter_context.empty():
        for i in event.parameter_context.lattice.min:
            bounds_string += str(i)
        bounds_string+='-'
        for i in event.parameter_context.lattice.max:
            bounds_string += str(i)
    else:
        bounds_string = 'none'
    marking_string = ''
    for i in range(0, len(graph.nodes)):
        if (event.marking[i] > 0):
            marking_string += (',' + graph.nodes[i].name + str(event.marking[i]))
    marking_string = marking_string[1:]
    #output.write('e' + str(event.id) + ' [label="' + event.target.name + str(event.target_value) + '{' + context_string + '}' + '(e' + str(event.id) + ')' + bounds_string + '{' + marking_string + '}" shape=box')
    output.write('e' + str(event.id) + ' [label="' + event.target.name + str(
        event.target_value) + '{' + context_string + '}' + '(e' + str(
        event.id) + ')' + '{' + marking_string + '}" shape=box')
    if event.cutoff:
        cutoff_count += 1
        output.write(' style=dashed')
    if event.goal:
        output.write(' color=crimson')
    output.write('];\n')
    for condition in event.preset:
        output.write('c' + str(condition.id) + ' -> e' + str(event.id) + ';\n')
    for condition in event.poset:
        output.write('e' + str(event.id) + ' -> c' + str(condition.id) + ';\n')

output.write('graph [label="' + str(event_count) + ' events (' + str(event_count - cutoff_count) + '/' + str(cutoff_count) + ' non-/cutoff)"]\n}')

output.close()

if timed:
    exec_time = time.clock() - exec_time
    print('Execution time: ' + '{0:.6f}'.format(exec_time))