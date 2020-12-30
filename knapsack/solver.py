from collections import namedtuple
import numpy as np
import sys
import numba as nb
from gurobipy import *
from psutil import cpu_count


@nb.njit()
def dynprog(item_count, capacity, values, weights):
    table = np.zeros((item_count + 1, capacity + 1), dtype=np.int32)
    for i in range(item_count):
        for w in range(capacity + 1):
            if weights[i] <= w:
                p1 = table[i][w]
                p2 = values[i] + table[i][w - weights[i]]
                table[i + 1][w] = max(p1, p2)
            else:
                table[i + 1][w] = table[i][w]

    knapsack = []
    knapsack_opt_val = table[-1][-1]  # [-1][-1]
    w = capacity
    for i in range(item_count, 0, -1):
        if table[i][w] != table[i - 1][w]:
            knapsack.append(1)
            w -= weights[i-1]
            i -= 1
        else:
            knapsack.append(0)

    return (knapsack_opt_val,1,knapsack)

def solve_it_np(input_data):
    lines = input_data.split('\n')

    firstLine = lines[0].split()
    item_count = int(firstLine[0])
    capacity = int(firstLine[1])

    items = []
    values = np.zeros((item_count), dtype=np.int32)
    weights = np.zeros((item_count), dtype=np.int32)

    for i in range(1, item_count+1):
        line = lines[i]
        parts = line.split()
        values[i-1] = int(parts[0])
        weights[i-1] = int(parts[1])

    knapsack = dynprog(item_count, capacity, values, weights)
    knapsack[2].reverse()
    output_data = f"{knapsack[0]} {knapsack[1]}\n" + " ".join(map(str, knapsack[2]))

    return output_data

def solve_it(input_data):


    lines = input_data.split('\n')

    firstLine = lines[0].split()
    item_count = int(firstLine[0])
    capacity = int(firstLine[1])

    values = [0] * item_count
    weights = [0] * item_count
    for i in range(1, item_count + 1):
        line = lines[i]
        parts = line.split()
        values[i - 1] = int(parts[0])
        weights[i - 1] = int(parts[1])

    m = Model("knapsack")
    m.setParam('OutputFlag', False)
    m.setParam("Threads", cpu_count())

    x = m.addVars(item_count, vtype=GRB.BINARY, name="items")
    m.setObjective(LinExpr(values, [x[i] for i in range(item_count)]), GRB.MAXIMIZE)
    m.addLConstr(LinExpr(weights, [x[i] for i in range(item_count)]), GRB.LESS_EQUAL, capacity, name="capacity")

    m.update()
    m.optimize()

    if m.status == 2:
        opt = 1
    else:
        opt = 0

    #return int(m.objVal), opt, [int(var.x) for var in m.getVars()]
    output_data = f"{int(m.objVal)} {opt}\n" + " ".join(map(str, [int(var.x) for var in m.getVars()]))
    return output_data

def solve_it_mip(input_data):
    from mip import Model, xsum, maximize, BINARY

    lines = input_data.split('\n')

    firstLine = lines[0].split()
    item_count = int(firstLine[0])
    capacity = int(firstLine[1])
    I = range(item_count)
    knapsack = [0] * item_count

    values = [0] * item_count
    weights = [0] * item_count

    for i in range(1, item_count + 1):
        line = lines[i]
        parts = line.split()
        values[i - 1] = int(parts[0])
        weights[i - 1] = int(parts[1])

    m = Model("knapsack")

    x = [m.add_var(var_type=BINARY) for i in I]

    m.objective = maximize(xsum(values[i] * x[i] for i in I))

    m += xsum(weights[i] * x[i] for i in I) <= capacity

    m.verbose = 0
    #m.emphasis = 1
    status = m.optimize()

    selected = [i for i in I if x[i].x >= 0.99]
    for index in selected:
        knapsack[index] = knapsack[index] + 1
    output_data = f"{int(m.objective_value)} {1}\n" + " ".join(map(str, knapsack))
    return output_data

if __name__ == "__main__":
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, "r") as input_data_file:
            input_data = input_data_file.read()
            # print(solve_it_np(input_data))
            print(solve_it(input_data))





