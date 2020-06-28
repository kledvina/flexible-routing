### sdvrp.py
### this file solves the split demand vrp


#-------------------------------------------------------------------------

def solve_OR_Tools_VRP(customers, depots, Q):
    """Solve the CVRP problem."""

    def create_distance_matrix(nodes):
        """Createss pairwise distance matrix."""
        distance_matrix = []
        for i in nodes:
            distance_row = []
            for j in nodes:
                distance_row.append(math.sqrt((nodes[i]. x -nodes[j].x )* *2 + (nodes[i]. y -nodes[j].y )* *2))
            distance_matrix.append(distance_row)
        return distance_matrix

    def create_data_model(customers, depots, Q):
        """Stores the data for the problem."""
        # Create set of nodes with both depots and customers
        nodes = {}
        nodes.update(depots)
        nodes.update(customers)
        data = {}
        # data['locations'] = [nodes[i].loc for i in nodes]
        data['distance_matrix'] = create_distance_matrix(nodes)
        data['demands'] = [0] + [customers[c].d for c in customers]
        data['vehicle_capacities'] = [Q ] *len(customers)
        data['num_vehicles'] = len(customers)
        data['depot'] = 0
        return data


    def print_solution(data, manager, routing, solution):
        """Prints solution on console."""
        total_distance = 0
        total_load = 0
        for vehicle_id in range(data['num_vehicles']):
            index = routing.Start(vehicle_id)
            plan_output = 'Route for vehicle {}:\n'.format(vehicle_id)
            route_distance = 0
            route_load = 0
            while not routing.IsEnd(index):
                node_index = manager.IndexToNode(index)
                route_load += data['demands'][node_index]
                plan_output += ' {0} Load({1}) -> '.format(node_index, route_load)
                previous_index = index
                index = solution.Value(routing.NextVar(index))
                route_distance += routing.GetArcCostForVehicle(
                    previous_index, index, vehicle_id)
            plan_output += ' {0} Load({1})\n'.format(manager.IndexToNode(index),
                                                     route_load)
            plan_output += 'Distance of the route: {}m\n'.format(route_distance)
            plan_output += 'Load of the route: {}\n'.format(route_load)
            print(plan_output)
            total_distance += route_distance
            total_load += route_load
        print('Total distance of all routes: {}m'.format(total_distance))
        print('Total load of all routes: {}'.format(total_load))

    def get_routes(solution, routing, manager):
        """Get vehicle routes from a solution and store them in an array."""
        # Get vehicle routes and store them in a two dimensional array whose
        # i,j entry is the jth location visited by vehicle i along its route.
        routes = []
        for route_nbr in range(routing.vehicles()):
            index = routing.Start(route_nbr)
            route = [manager.IndexToNode(index)]
            while not routing.IsEnd(index):
                index = solution.Value(routing.NextVar(index))
                route.append(manager.IndexToNode(index))
            routes.append(route)
        return routes

    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['distance_matrix'][from_node][to_node]

    def demand_callback(from_index):
        """Returns the demand of the node."""
        # Convert from routing variable Index to demands NodeIndex.
        from_node = manager.IndexToNode(from_index)
        return data['demands'][from_node]

    ###################
    ### RUN PROGRAM ###
    ###################
    # Return 0 cost if no demands
    if all([customers[cust].d == 0 for cust in customers]):
        return (0, 0, 0)

    start = time.time()

    # Instantiate the data problem.
    data = create_data_model(customers, depots, Q)

    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']),
                                           data['num_vehicles'], data['depot'])
    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)

    # Create and register a transit callback.
    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    # Define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Add Capacity constraint.
    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,  # null capacity slack
        data['vehicle_capacities'],  # vehicle maximum capacities
        True,  # start cumul to zero
        'Capacity')

    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

    # Solve the problem.
    solution = routing.SolveWithParameters(search_parameters)
    end = time.time()
    routes = get_routes(solution, routing, manager)
    # print(routes)
    num_trips = sum(sum(d for d in r ) >0 for r in routes)
    return solution.ObjectiveValue(), en d -start, num_trips
    """
    # Print solution on console.
    if solution:
        print('Solution found.')
        print('Objective: {}'.format(solution.ObjectiveValue()))
        #print_solution(data, manager, routing, solution)
        for i, route in enumerate(routes):
            print('Route', i, route)
    """