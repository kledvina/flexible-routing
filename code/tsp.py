### tsp.py
### this file generates a traveling salesman tour


#-------------------------------------------------------------------------

def solve_Concorde_TSP(customers):

    x_data = [customers[c].x for c in customers]
    y_data = [customers[c].y for c in customers]
    solver = TSPSolver.from_data(x_data, y_data, 'EUC_2D')
    start = time.time()
    solution = solver.solve()
    end = time.time()
    obj = solution.optimal_value
    solution_IDs = [list(customers.keys())[x] for x in solution.tour]
    tau = dict([( i +1, customers[c]) for i ,c in zip(range(len(customers)), solution_IDs)])
    return obj, get_routing_solution(tau), en d -start

#-------------------------------------------------------------------------
