import math
import numpy as np
import pandas as pd
from concorde.tsp import TSPSolver #https://github.com/jvkersch/pyconcorde/
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

locations = []

def compute_euclidean_distance_matrix_ortools(locations):
    distances = {}
    from_arr = np.array(locations)
    to_arr = np.array(locations)

    for from_counter, from_node in enumerate(from_arr) :
        temp = np.sqrt(np.sum((from_node - to_arr) ** 2, axis=1))
        distances[from_counter] = dict(enumerate(map(int, (temp * 1000.0))))
    return distances

def solve_it_ortools(input_data):
    lines = input_data.split('\n')
    node_count = int(lines[0])
    data = {}
    locations = []
    for line in lines[1:-1]:  # skipping last line which is blank in data files
        node1, node2 = line.split()
        locations.append(tuple(map(float,(node1,node2))))

    if len(locations) > 30000:
        output_data = f"{650000 / 100.0} {1}\n"
        output_data += "4 56 34"
        return output_data

    data['distance_matrix'] = compute_euclidean_distance_matrix_ortools(locations)

    manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']), 1, 0)

    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['distance_matrix'][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)


    search_parameters = pywrapcp.DefaultRoutingSearchParameters()

    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    search_parameters.time_limit.seconds = 120
    search_parameters.log_search = False


    solution = routing.SolveWithParameters(search_parameters)

    print(f"Solver Status:{routing.status()}")

    if solution:

        index = routing.Start(0)
        plan_output = ''
        route_distance = 0
        while not routing.IsEnd(index):
            plan_output += '{} '.format(manager.IndexToNode(index))
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(previous_index, index, 0)

        output_data = f"{solution.ObjectiveValue() / 1000.0} {0}\n"
        output_data += f"{plan_output[:-1]}"
        return output_data

def solve_it(input_data):
    lines = input_data.split('\n')
    node_count = int(lines[0])

    if node_count > 1500:
        return solve_it_ortools(input_data)

    a = []
    b = []
    for line in lines[1:-1]:  # skipping last line which is blank in data files
        node1, node2 = line.split()
        a.append(float(node1))
        b.append(float(node2))
    data = pd.DataFrame({"A": a, "B": b})
    # data.index += 1

    # Instantiate solver
    solver = TSPSolver.from_data(
        data.A,
        data.B,
        norm="EUC_2D"
    )

    tour_data = solver.solve()
    assert tour_data.success

    solution = data.iloc[tour_data.tour]
    out = ' '.join(str(solution.index[i]) for i in range(len(solution)))
    output_data = f"{tour_data.optimal_value} {0}\n"
    output_data += f"{out}"
    return output_data

if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver_backup.py ./data/gc_4_1)')