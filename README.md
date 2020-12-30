# DiscreteOptimization

# My solutions for Discrete Optimization course on Coursera

This is undoubedly one of the toughest courses I have ever taken. 
The assignment problems get tougher, you will need to switch the solution strategy depending on the size of input data for the same assignnment,
so you will see that I use different solvers in the course of the same assignment.

I have used following solvers for each of the problems:

Knapsack problem - Gurobi

Graph Coloring problem - MiniZinc

Traveling Salesman problem - OR-Tools for small number of locations to visit, Concorde TSPSolver for large number of locations (the last problem in the assignment had 30,000 locations to visit which could not be solved by any of the solvers I tried)

Facilities Location problem - Gurobi (tooks 2 hours to finish last problem of 2000 facilities with 2000 customers)

Vehicle Routing problem - PySCIPOpt (required a different strategy for customer count > 20)

This is THE TOUGHEST problem of the course!!!

