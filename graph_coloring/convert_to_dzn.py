
def convert_data_to_dzn(input_data, dzn_filename = 'data.dzn'):
    '''Reformat input_data to a minizinc .dzn file, write to disk'''

    # parse the input
    lines = input_data.split('\n')
    first_line = lines[0].split()
    node_count = int(first_line[0])
    edge_count = int(first_line[1])

    with open(f'{dzn_filename}', 'w') as f:
        f.write(f"NUM_NODES = {node_count};\n")
        f.write(f"NUM_EDGES = {edge_count};\n")
        f.write(f"edges = [|")
        for line in lines[1:-1]:  # skipping last line which is blank in data files
            node1, node2 = line.split()
            f.write(f"\n{node1}, {node2}|")
        f.write(']\n')

    return node_count

if __name__ == '__main__':
    import sys
    import os

    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            filename = os.path.basename(file_location)
            input_data = input_data_file.read()
        print(convert_data_to_dzn(input_data, filename+".dzn"))
    else:
        print(
            'This test requires an input file.  Please select one from the data directory. (i.e. python solver_backup.py ./data/gc_4_1)')