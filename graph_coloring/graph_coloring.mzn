include "globals.mzn";

% Input Parameters (values for these specified in .dzn data file)
int: NUM_NODES; 
int: NUM_EDGES;
int: MAX_COLOR;
array[1..NUM_EDGES,1..2] of int: edges;

% Decision Variable: the solver will find the values of these
array[0..NUM_NODES-1] of var 1..NUM_NODES: color;

% Our Constraints 
constraint forall(e in 1..NUM_EDGES)(color[edges[e,1]] != color[edges[e,2]]);

% Symmetry-Breaking Constraint
constraint forall(n in 1..NUM_NODES-1)(value_precede(n, n+1, color));

% Our Objective Function
%solve minimize max(color);
constraint max(color) <= MAX_COLOR;
solve satisfy;

% formatted output
output[show(max(color)) ++ "\n" ++ show(color)];