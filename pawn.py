import sys
import time
import regulatory_network
import parametrised_unfolding

input_file_name = ""
output_file_name = ""

target = 0

timed = False
exec_time = 0

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
                target = sys.argv[i + 1]
                i += 1
            elif c == 'r':
                if i == (len(sys.argv) - 1):
                    print("The -r modifier is expected to be followed by a number.")
                    exit(2)
                parametrised_unfolding.report_frequency = int(sys.argv[i + 1])
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
    input_file_name = args[0];
    output_file_name = args[1];

if timed:
    exec_time = time.clock()

graph = regulatory_network.parse_regulatory_graph(input_file_name)

if target:
    parametrised_unfolding.target = graph.build_marking(target)
    if not parametrised_unfolding.target:
        print('The specified marking "' + target + '" does not match the system definition. Usage: Pawn <input file> [<output file>,-v,-g <target node>]')
        exit(2)

unf = parametrised_unfolding.unfold(graph)

output = open(output_file_name, 'w')

output.write("digraph {\n")

for c in unf.conditions:
    output.write('c' + str(c.id) + ' [label="' + c.node.name + str(c.value) + '(c' + str(c.id) + ')" shape=circle];\n')

ev_count = 0
co_count = 0
for e in unf.events:
    ev_count += 1
    context_string = ''
    for i in range(0, len(graph.nodes)):
        if (e.context.regulators[i] > 0):
            context_string += (',' + graph.nodes[i].name + str(e.context.regulators[i]))
    context_string = context_string[1:]
    hcstring = ''
    if not e.parameter_context.empty():
        for i in e.parameter_context.interval.min:
            hcstring += str(i)
        hcstring+='-'
        for i in e.parameter_context.interval.max:
            hcstring += str(i)
    else:
        hcstring = 'none'
    marking_string = ''
    for i in range(0, len(graph.nodes)):
        if (e.marking[i] > 0):
            marking_string += (',' + graph.nodes[i].name + str(e.marking[i]))
    marking_string = marking_string[1:]
    output.write('e' + str(e.id) + ' [label="' + e.target.name + str(e.target_value) + '{' + context_string + '}' + '(e' + str(e.id) + ')' + hcstring + '{' + marking_string + '}" shape=box')
    if e.cutoff:
        co_count += 1
        output.write(' style=dashed')
    if e.goal:
        output.write(' color=crimson')
    output.write('];\n')
    for c in e.preset:
        output.write('c' + str(c.id) + ' -> e' + str(e.id) + ';\n')
    for c in e.poset:
        output.write('e' + str(e.id) + ' -> c' + str(c.id) + ';\n')

output.write('graph [label="' + str(ev_count) + ' events (' + str(ev_count - co_count) + '/' + str(co_count) + ' non-/cutoff)"]\n}')

output.close()

if timed:
    exec_time = time.clock() - exec_time
    print('Execution time: ' + '{0:.6f}'.format(exec_time))