import subprocess
from convert_to_dzn import convert_data_to_dzn
import pandas as pd

orig_filename = None

def solve_it(input_data):

    # Run input_data through Minizinc Model
    mz_model = "graph_coloring.mzn"
    node_count = convert_data_to_dzn(input_data, dzn_filename = f'{orig_filename}.dzn')

    if node_count == 50:
        # Problem 1
        MAX_COLOR=6 #5 #7 #10
        SOLVER='Default'
    elif node_count == 70:
        # Problem 2
        MAX_COLOR=17 #18 #19 #20
        SOLVER = 'Default'
    elif node_count == 100:
        # Problem 3
        MAX_COLOR=17 #16 #18 #19 #20
        SOLVER='Default'
    elif node_count == 250:
        # Problem 4
        MAX_COLOR=93 #94 #95 #100
        SOLVER = 'Default'
    elif node_count == 500:
        # Problem 5
        MAX_COLOR=15 #14 #16 #18 #19
        SOLVER = 'Default'
    elif node_count == 1000:
        # Problem 6
        MAX_COLOR=122 #123 #300
        SOLVER = 'Default'

    verbose=True
    statistics=True
    args = ["C:\\MiniZinc\\minizinc.exe", mz_model, f"{orig_filename}.dzn", '-D', 'MAX_COLOR=' + str(MAX_COLOR), '-p', '8']

    ## OR-Tools did not work for any of the assignment problems, the default Gecode was much better
    if SOLVER == 'OR-Tools':
        args.append('--solver')
        args.append('com.google.or-tools')
    if verbose:
        args.append('--verbose')
    if statistics:
        pass
        #args.append('--statistics')
    start = pd.Timestamp.utcnow()
    mz_output = subprocess.run(
        [ str(x) for x in args ],
        shell=False, stdout=subprocess.PIPE).stdout.decode('utf-8')
    print(f"finished minizinc run in {pd.Timestamp.utcnow()-start}")

    # Parse Minizinc Output and Prepare Coursera Course Output
    OPTIMAL = 0
    if "==========" in mz_output:
        OPTIMAL = 1

    colors = mz_output.split('\r\n')[1].strip('[]').split(', ')
    #num_colors = len(set(colors))
    num_colors = mz_output.split('\r\n')[0]

    output_data = f"{num_colors} {OPTIMAL}\n"
    output_data += ' '.join(colors)

    return output_data

if __name__ == '__main__':
    import sys
    import os
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            orig_filename = os.path.basename(file_location)
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver_backup.py ./data/gc_4_1)')


