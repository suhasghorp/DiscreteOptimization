#!/usr/bin/python
# -*- coding: utf-8 -*-

import math
import networkx
from collections import namedtuple, OrderedDict
from pyscipopt import Model, quicksum, multidict, Conshdlr, SCIP_RESULT, SCIP_PRESOLTIMING, SCIP_PROPTIMING
from datetime import datetime
Customer = namedtuple("Customer", ['index', 'demand', 'x', 'y'])

def length(customer1, customer2):
    return math.sqrt((customer1.x - customer2.x)**2 + (customer1.y - customer2.y)**2)

def dist_between_nodes(node_to_check, nodes, c):

    sum = 0
    for node in nodes:
        sum += c[(node_to_check[0], node[0])]

    return sum

def cost_of_new_customer_in_vehicle(c, i, w):
    return min(c[(0, i)] + c[(i, w)] + c[(w, 0)], c[(0, w)] + c[(w, i)] + c[(i, 0)]) - (c[(0, w)] + c[w, 0])

def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    parts = lines[0].split()
    customer_count = int(parts[0])
    vehicle_count = int(parts[1])
    vehicle_capacity = int(parts[2])
    
    customers = []
    for i in range(1, customer_count+1):
        line = lines[i]
        parts = line.split()
        customers.append(Customer(i-1, int(parts[0]), float(parts[1]), float(parts[2])))

    #the depot is always the first customer in the input
    depot = customers[0]

    obj = None

    if customer_count <= 20:
        obj, vehicle_tours = scip_solver_2(customers, customer_count, vehicle_count, vehicle_capacity)

    else:
        obj, vehicle_tours = scip_solver_4(customers, customer_count, vehicle_count, vehicle_capacity)

    if obj:
        # prepare the solution in the specified output format
        outputData = '%.2f' % obj + ' ' + str(0) + '\n'
        if len(vehicle_tours) > 0:
            for v in range(0, vehicle_count):
                outputData += str(depot.index) + ' ' + ' '.join(
                    [str(customer.index) for customer in vehicle_tours[v]]) + ' ' + str(depot.index) + '\n'

        return outputData

    return None

def scip_solver_2(customers, customer_count, vehicle_count, vehicle_capacity):
    model = Model("vrp")

    c_range = range(0, customer_count)
    cd_range = range(1, customer_count)

    x, d, w, v = {}, {}, {}, {}
    for i in c_range:
        for j in c_range:
            d[i, j] = length(customers[i], customers[j])
            w[i, j] = customers[i].demand + customers[j].demand
            if j > i and i == 0:  # depot
                x[i, j] = model.addVar(ub=2, vtype="I", name="x(%s,%s)" % (i, j))
            elif j > i:
                x[i, j] = model.addVar(ub=1, vtype="I", name="x(%s,%s)" % (i, j))

    model.addCons(quicksum(x[0, j] for j in cd_range) <= 2 * vehicle_count, "DegreeDepot")

    for i in cd_range:
        model.addCons(quicksum(x[j, i] for j in c_range if j < i) +
                      quicksum(x[i, j] for j in c_range if j > i) == 2, "Degree(%s)" % i)



    model.setObjective(quicksum(d[i, j] * x[i, j] for i in c_range for j in c_range if j > i), "minimize")

    mip_gaps = [0.0]
    runs = 0

    start = datetime.now()
    for gap in mip_gaps:
        model.freeTransform()
        model.setRealParam("limits/gap", gap)
        model.setRealParam("limits/time", 60 * 20)  # Time limit in seconds
        edges, final_edges, runs = optimize(customer_count, customers, model, vehicle_capacity, vehicle_count, x)

    run_time = datetime.now() - start

    print(edges)
    print(final_edges)
    output = [[]] * vehicle_count
    for i in range(vehicle_count):
        output[i] = []
        current_item = None
        if len(final_edges) > 0:

            for e in final_edges:
                if e[0] == 0:
                    current_item = e
                    break
            if current_item:
                a = current_item[0]
                current_node = current_item[1]
                output[i].append(customers[current_node])
                final_edges.remove(current_item)
                searching_connections = True
                while searching_connections and len(final_edges) > 0:
                    for edge in final_edges:
                        found_connection = False
                        a_edge = edge[0]
                        b_edge = edge[1]


                        if b_edge == current_node and a_edge == 0:
                            final_edges.remove(edge)
                            break

                        if a_edge == current_node:
                            output[i].append(customers[b_edge])
                            current_node = b_edge
                            found_connection = True
                        elif b_edge == current_node:
                            output[i].append(customers[a_edge])
                            current_node = a_edge
                            found_connection = True

                        if found_connection:
                            final_edges.remove(edge)
                            break

                    if not found_connection:
                        searching_connections = False

    print(output)

    sol = model.getBestSol()
    obj = model.getSolObjVal(sol)

    print("RUN TIME: %s" % str(run_time))
    print("NUMBER OF OPTIMIZATION RUNS: %s" % runs)

    return obj, output

def optimize(customer_count, customers, model, vehicle_capacity, vehicle_count, x, cut=True):
    EPS = 1.e-6
    runs = 0
    while True:
        print("Solving problem with %s customers and %s vehicles..." % (customer_count, vehicle_count))
        model.optimize()
        runs += 1
        sol = model.getBestSol()
        final_edges = []
        edges = []
        for (i, j) in x:
            val = model.getSolVal(sol, x[i, j])
            if val > EPS:
                if i != 0 and j != 0:
                    edges.append((i, j))
                final_edges.append((i, j))

        if not cut:
            break

        if not addcut(edges, model, customers, vehicle_capacity, x):
            break
    return edges, final_edges, runs

def addcut(cut_edges, model, customers, vehicle_capacity, x):

    G = networkx.Graph()
    G.add_edges_from(cut_edges)
    Components = networkx.connected_components(G)
    cut = False
    model.freeTransform()
    for S in Components:
        S_card = len(S)
        q_sum = sum(customers[i].demand for i in S)
        NS = int(math.ceil(float(q_sum) / vehicle_capacity))
        S_edges = [(i, j) for i in S for j in S if i < j and (i, j) in cut_edges]
        if S_card >= 3 and (len(S_edges) >= S_card or NS > 1):

            model.addCons(quicksum(x[i, j] for i in S for j in S if j > i) <= S_card - NS)

            cut = True
    return cut

def scip_solver_4(customers, customer_count, vehicle_count, vehicle_capacity):

    K = vehicle_count
    n = customer_count
    b = vehicle_capacity
    a = {}  # the demand for customer i
    c = {}  # the distance between customer i and j

    for i, x in enumerate(customers):
        a[i] = x.demand

        for j, y in enumerate(customers):
            # if i != j:
            c[(i, j)] = length(x, y)

    best_obj = 99999999
    best_tours = []

    ### VEHICLE CAPACITY SUB-PROBLEM ###
    obj, vehicle_tours = vehicle_assignment_solver(K, a, b, c, customer_count)

    ### TSP ON VEHICLES SUB-PROBLEM ###
    final_obj, final_tours = tsp_solver(c, customers, vehicle_tours)

    if final_obj < best_obj and len([item for elem in final_tours for item in elem]) == n - 1:
        best_obj = final_obj
        best_tours = final_tours
        print("Best solution found: %s" % best_obj)

    print("Best objective cost: %s" % best_obj)

    return best_obj, best_tours

def vehicle_assignment_solver(K, a, b, c, customer_count, predefined_vehicle_index=None, predefined_vehicle_nodes=None):
    v_range = range(0, K)
    c_range = range(1, customer_count)
    samples = []


    a_ord = sorted(a.items(), key=lambda k: k[1])
    a_ord.reverse()
    samples.append(a_ord[0])
    del a_ord[0]

    while True:
        a_ord = sorted(a_ord, key=lambda el: dist_between_nodes(el, samples, c), reverse=True)
        samples.append(a_ord[0])
        del a_ord[0]

        if len(samples) == K:
            break

    w = {}
    for i in v_range:
        w[i] = samples[i][0]

    # Resolve MIP problem to find the best possible vehicle-customers combinations
    model = Model("vrp_vehicles")
    #model.hideOutput()

    if customer_count >= 200:
        model.setRealParam("limits/gap", 0.005)


    y = {}
    for v in v_range:
        for i in c_range:
            # customer i is visited by vehicle v
            y[i, v] = model.addVar(vtype="B", name="y(%s,%s)" % (i, v))
    for v in v_range:
        # Constraint: the demand of customers assigned to vehicle V cannot exceed its capacity
        model.addCons(quicksum(a[i] * y[i, v] for i in c_range) <= b)

        # Constraint: for this model, we enforce every vehicle to visit a customer
        # model.addCons(quicksum(y[i, v] for i in c_range) >= 1)

        # Constraint: we enforce the customers on 'w' to be visited by the defined vehicle
        #model.addCons(y[w[v], v] == 1)

        #if predefined_vehicle_nodes and v == predefined_vehicle_index:
        #    for p in predefined_vehicle_nodes:
        #        model.addCons(y[p, predefined_vehicle_index] == 1)

    for i in c_range:
        # if i > 0:
        # Constraint: each customer has to be visited by exactly one vehicle
        model.addCons(quicksum(y[i, v] for v in v_range) == 1)

    model.setObjective(quicksum(quicksum(cost_of_new_customer_in_vehicle(c, i, w[v])*y[i, v]
                                         for i in c_range) for v in v_range), "maximize")

    model.optimize()
    # best_sol = model.getBestSol()
    vehicle_tours = {}
    for v in v_range:
        vehicle_tours[v] = []
        for i in c_range:
            # val = model.getSolVal(best_sol, y[i, v])
            val = model.getVal(y[i, v])
            if val > 0.5:
                vehicle_tours[v].append(i)

    obj = model.getObjVal()

    return obj, vehicle_tours

def tsp_solver(c, customers, vehicle_tours):
    def addcut(cut_edges):
        G = networkx.Graph()
        G.add_edges_from(cut_edges)
        Components = list(networkx.connected_components(G))
        if len(Components) == 1:
            return False
        model.freeTransform()
        for S in Components:
            model.addCons(quicksum(x[i, j] for i in S for j in S) <= len(S) - 1)

        return True

    # Add the depot on each vehicle
    vehicle_tours = {k: vehicle_tours[k] + [0] for k in vehicle_tours.keys()}
    final_obj = 0
    final_tours = []
    for key, value in vehicle_tours.items():
        v_customers = value
        model = Model("vrp_tsp")

        x = {}

        for i in v_customers:
            for j in v_customers:
                # vehicle moves from customer i to customer j
                x[i, j] = model.addVar(vtype="B", name="x(%s,%s)" % (i, j))

        for i in v_customers:
            # Constraint: every customer can only be visited once
            # (or, every node must be connected and connect to another node)
            model.addCons(quicksum(x[i, j] for j in v_customers) == 1)
            model.addCons(quicksum(x[j, i] for j in v_customers) == 1)

            for j in v_customers:
                if i == j:
                    # Constraint: a node cannot conect to itself
                    model.addCons(x[i, j] == 0)

        # Objective function: minimize total distance of the tour
        model.setObjective(quicksum(x[i, j] * c[(i, j)] for i in v_customers for j in v_customers), "minimize")

        EPS = 1.e-6
        isMIP = False
        while True:
            model.optimize()
            edges = []
            for (i, j) in x:
                if model.getVal(x[i, j]) > EPS:
                    edges.append((i, j))

            if addcut(edges) == False:
                if isMIP:  # integer variables, components connected: solution found
                    break
                model.freeTransform()
                for (i, j) in x:  # all components connected, switch to integer model
                    model.chgVarType(x[i, j], "B")
                    isMIP = True

        # model.optimize()
        best_sol = model.getBestSol()
        sub_tour = []

        # Build the graph path
        # Retrieve the last node of the graph, i.e., the last one connecting to the depot
        last_node = [n for n in edges if n[1] == 0][0][0]
        G = networkx.Graph()
        G.add_edges_from(edges)
        path = list(networkx.all_simple_paths(G, source=0, target=last_node))
        path.sort(reverse=True, key=lambda u: len(u))

        if len(path) > 0:
            path = path[0][1:]
        else:
            path = path[1:]

        obj = model.getSolObjVal(best_sol)
        final_obj += obj
        final_tours.append([customers[i] for i in path])


    return final_obj, final_tours

import sys

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:

        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/vrp_5_4_1)')

