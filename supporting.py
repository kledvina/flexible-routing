import numpy as np
import math
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

# GLOBAL VARIABLES
field_width = 100 # Customer location has x-coordinate in (0, field_width)
field_height = 100 # Customer location has y-coordinate in (0, field_height)
#depot_x = 50 # Depot x-coordinate
#depot_y = 50 # Depot y-coordinate

#---------------------------------------------------------------------------------

class Instance():
    """A realized set of node locations and demands and the resulting routing characteristics."""

    def __init__(self, xlocs, ylocs, demands, solve_TSP=True):

        self.size = len(demands) - 1
        self.demands = demands
        self.xlocs = xlocs
        self.ylocs = ylocs
        self.distances = self.calc_distance_matrix()
        self.optimal_routes = 'None'
        self.tour = 'None'
        if solve_TSP:
            # self.tour = self.solve_TSP()
            # self.tour = self.TSP_heuristic()
            self.tour = self.solve_TSP()

    def calc_distance_matrix(self):
        """Returns a matrix with pairwise node distances"""
        distances = np.zeros((self.size + 1, self.size + 1), dtype=float)
        for i in range(self.size + 1):
            for j in range(self.size + 1):
                new_dist = math.sqrt((self.xlocs[i] - self.xlocs[j]) ** 2 + (self.ylocs[i] - self.ylocs[j]) ** 2)
                distances[i, j] = new_dist
        return distances

    def update_demands(self, demands):
        self.demands = demands

    def update_tour(self, tour):
        self.tour = tour

    def get_lowerbound(self, capacity):
        """Returns a theoretical lowerbound on the optimal routing cost"""
        return (2 / capacity) * sum([self.demands[i] * self.distances[0, i]
                                     for i in range(len(self.demands))])

    def get_fleet_size(self, route_size):
        """Returns the number of vehicles needed to visit all nodes given a fixed route size"""
        assert self.size % route_size == 0, "Number of customers must be evenly divisible by the route size."
        return int(self.size / route_size)

    def save_optimal_routes(self, route_list):
        self.optimal_routes = route_list

    def solve_TSP(self):

        def create_data_model():
            data = {}
            data['distance_matrix'] = self.distances
            data['num_vehicles'] = 1
            data['depot'] = 0
            return data

        def get_tour(manager, routing, solution):
            index = routing.Start(0)
            plan_output = ''
            while not routing.IsEnd(index):
                plan_output += '{},'.format(manager.IndexToNode(index))
                previous_index = index
                index = solution.Value(routing.NextVar(index))
            plan_output += '{}'.format(manager.IndexToNode(index))
            as_list = plan_output.split(',')
            return [int(i) for i in as_list][:-1]  # excludes return to depot in tour

        # --- RUN PROGRAM ---

        # Instantiate the data problem.
        data = create_data_model()

        # Create the routing index manager.
        manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']),
                                               data['num_vehicles'], data['depot'])

        # Create Routing Model.
        routing = pywrapcp.RoutingModel(manager)

        def distance_callback(from_index, to_index):
            """Returns the distance between the two nodes."""
            # Convert from routing variable Index to distance matrix NodeIndex.
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return data['distance_matrix'][from_node][to_node]

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)

        # Define cost of each arc.
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        # Setting first solution heuristic.
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

        # Solve the problem.
        solution = routing.SolveWithParameters(search_parameters)
        return get_tour(manager, routing, solution)


#---------------------------------------------------------------------------------

def get_trip_count(route_list):
    """Returns number of trips in route list"""
    assert type(route_list[0]) == list, "route_list must be a list of lists (routes)"
    count = 0
    for route in route_list:
        if route != []:
            count += 1
    return count


#---------------------------------------------------------------------------------

def get_circular_cost(inst, segment):
    """Returns the total distance of moving from node to node within the given segment"""
    if len(segment) == 0:
        return 0
    else:
        return sum([inst.distances[segment[i], segment[i + 1]] for i in range(len(segment) - 1)])


#---------------------------------------------------------------------------------

def get_radial_cost(inst, segment):
    """Returns the total distance of trips to/from the depot at segment endpoints."""
    if len(segment) == 0:
        return 0
    else:
        return inst.distances[0, segment[0]] + inst.distances[0, segment[-1]]


#---------------------------------------------------------------------------------

def get_total_cost(inst, segment):
    """Returns sum of circular and radial costs for the given segment"""
    return get_circular_cost(inst, segment) + get_radial_cost(inst, segment)


#---------------------------------------------------------------------------------

def optimize(inst, capacity):
    """Note: This function is largely from the Google OR-Tools tutorial."""

    def create_data_model(inst, capacity):
        data = {}
        data['distance_matrix'] = inst.distances
        data['demands'] = inst.demands
        data['vehicle_capacities'] = [capacity]*inst.size
        data['num_vehicles'] = sum(inst.demands)
        data['depot'] = 0
        return data

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
    
    
    # --- RUN PROGRAM ---

    # Zero cost if no demands
    if all(dem == 0 for dem in inst.demands):
        return [[]]

    # Set up data model
    data = create_data_model(inst, capacity)
    
    # Create the routing index manager
    manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']), data['num_vehicles'], data['depot'])
    
    # Create routing model
    routing = pywrapcp.RoutingModel(manager)

    # Create and register a transit callback
    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    # Define cost of each arc
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Add capacity constraint
    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,  # null capacity slack
        data['vehicle_capacities'],  # vehicle maximum capacities
        True,  # start cumul to zero
        'Capacity')

    # Setting first solution heuristic
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

    # Solve the problem
    solution = routing.SolveWithParameters(search_parameters)
    all_routes = get_routes(solution, routing, manager)
    nonempty_routes = [route for route in all_routes if not all(i == 0 for i in route)]
    
    # Remove the depot from the optimal routes
    parsed_routes = [route[1:-1] for route in nonempty_routes]
    
    return parsed_routes
    
def solve_SDVRP(inst, capacity):
    """Creates equivalent demand/location instance with unit demand and solves the VRP with splittable demands"""
    # Create equivalent instance with unit demand customers
    depot_x = inst.xlocs[0]
    depot_y = inst.ylocs[0]
    
    split_xlocs = [[depot_x]] + [[inst.xlocs[i]]*inst.demands[i] for i in range(1,len(inst.demands))]
    split_xlocs = [v for sublist in split_xlocs for v in sublist]

    split_ylocs = [[depot_y]] + [[inst.ylocs[i]]*inst.demands[i] for i in range(1,len(inst.demands))]
    split_ylocs = [v for sublist in split_ylocs for v in sublist]

    split_demands = [[0]] + [[1]*inst.demands[i] for i in range(1,len(inst.demands))]
    split_demands = [v for sublist in split_demands for v in sublist]
    
    split_inst = Instance(split_xlocs, split_ylocs, split_demands, solve_TSP=False)

    # Solve VRP with unit demand customers
    vrp = optimize(split_inst, capacity)

    # Convert back to non-unit demand problem
    # to get SDVRP solution for original instance
    ids = [[i]*inst.demands[i] for i in range(1,len(inst.demands))]
    ids = [v for sublist in ids for v in sublist]
    if ids == []:
        return [[]] # No routes
    else:
        sdvrp = [[ids[c-1] for c in route] for route in vrp]
   
    return sdvrp

def solve_VRP(inst, capacity):
    depot_x = inst.xlocs[0]
    depot_y = inst.ylocs[0]
    # Create equivalent instance EXCLUDING customers with zero demand
    xlocs = [depot_x] + [inst.xlocs[i] for i in range(1,len(inst.demands)) if inst.demands[i] != 0]
    ylocs = [depot_y] + [inst.ylocs[i] for i in range(1,len(inst.demands)) if inst.demands[i] != 0]
    dems = [0] + [inst.demands[i]  for i in range(1,len(inst.demands)) if inst.demands[i] != 0]
    
    new_inst = Instance(xlocs, ylocs, dems, solve_TSP=False)

    # Solve VRP
    vrp = optimize(new_inst, capacity)
    return vrp


#---------------------------------------------------------------------------------

def get_primary_routes(inst, route_size):
    """Splits customer sequence into segments of 'route_size' number of customers.
    Requires that number of customers is evenly divisible by route_size."""
    
    assert inst.size%route_size == 0, "The number of customers must be evenly divisible by route_size."
    tour =  inst.tour[1:] # Exclude depot
    routes = []
    for i in range(0,len(tour),route_size):
        new_route = tour[i:min(i+route_size,len(tour))]
        routes.append(new_route)
    return routes


#---------------------------------------------------------------------------------

def get_extended_routes(inst, route_size, overlap_size):
    """Splits customer sequnce into segments of 'route_size + overlap_size' number of customers, where adjacent
    segments SHARE overlap_size number of customers. Requires that the number of customers is evenly divisible by route_size."""
    
    assert inst.size%route_size == 0, "The number of customers must be evenly divisible by primary route size."
    tour = inst.tour[1:]
    routes = []
    for i in range(0,len(tour),route_size):
        new_route = tour[i:min(i+route_size+overlap_size,len(tour))] # note: for the last route, subsetting ends with final customer
        routes.append(new_route)
    return routes


#---------------------------------------------------------------------------------

def create_full_trips(inst, route_list, capacity, demand_filled = None):
    """Splits a sequence of customers into individual trips. Returns a list of lists."""
    
    assert type(route_list[0]) == list, "route_list must be a list of lists (routes)."

    # Dictionary for tracking remaining demand filled at all customers
    remaining_demand = dict([(inst.tour[i],inst.demands[inst.tour[i]]) for i in range(1,len(inst.tour))])

    segments = []
    for m in range(len(route_list)):
        i = 0
        seg_dict = {} # demand filled on current trip
        vehicle_dict = dict([(inst.tour[i],0) for i in range(1,len(inst.tour))]) # total demand filled by vehicle on this route
        while i < len(route_list[m]):
            cust = route_list[m][i]
            for d in range(inst.demands[cust]):
                #print(dict([(c,vehicle_dict[c]) for c in vehicle_dict if vehicle_dict[c]!=0]))
                #print(dict([(c,seg_dict[c]) for c in seg_dict if seg_dict[c]!=0]))
                
                if demand_filled != None and sum(vehicle_dict.values()) == demand_filled[m]:
                    # Route's vehicle achieved its predetermined workload (if applicable)
                    # Force to end this route and move to next
                    i = len(route_list[m])
                    break
                    
                elif sum(remaining_demand[c] for c in route_list[m]) == 0:
                    # Route is completed
                    # Force to end this route and move to next
                    i = len(route_list[m])
                    break
                
                elif sum(seg_dict.values()) == capacity:
                    # Vehicle is at capacity
                    # End current trip, and begin a new trip within this route
                    segments.append(list(seg_dict))
                    seg_dict = {cust: 1}
                    vehicle_dict[cust] += 1
                    remaining_demand[cust] -= 1
                    
                elif remaining_demand[cust] > 0:
                    if cust not in seg_dict:
                        # Begin service
                        seg_dict[cust] = 1 
                    else:
                        # Continue service
                        seg_dict[cust] += 1
                    vehicle_dict[cust] += 1
                    remaining_demand[cust] -= 1
                
            i+=1 # Moves to next customer
        
        # Append route's last segment
        segments.append(list(seg_dict))

    return segments


#---------------------------------------------------------------------------------

def implement_k_overlapped_alg(inst, primary_routes, extended_routes, capacity, route_size, overlap_size):
    """Implement's general k-overlapped routing algorithm. Returns list of realized vehicle routes. """
    assert type(primary_routes[0]) == list, "primary_routes must be a list of lists (routes)"
    assert type(extended_routes[0]) == list, "extended_routes must be a list of lists (routes)"
    
    if overlap_size == 0: # reduced to dedicated routing
        return create_full_trips(inst,primary_routes,capacity)

    # Get overlapped segments (note that last route does not have any shared customers at the route's end)
    overlapped_segments = []
    for j in range(len(primary_routes) - 1):
        new_segment = [c for c in extended_routes[j] if c not in primary_routes[j]]
        overlapped_segments.append(new_segment)

    # Initialize arrays
    primary_demands = np.asarray([sum(inst.demands[cust] for cust in route) for route in
                                  primary_routes])  # a priori primary route demand for each vehicle
    extended_demands = np.asarray([sum(inst.demands[cust] for cust in route) for route in
                                   extended_routes])  # a priori extended route demand for each vehicle
    overlap_demands = extended_demands - primary_demands  # demands of customers in k-overlapped regions for each vehicle

    first = np.asarray([route[0] for route in primary_routes])  # first customer in route for each vehicle
    last = np.asarray([route[-1] for route in overlapped_segments] + [inst.tour[-1]])

    excess = np.zeros(len(primary_routes))  # surplus capacity for each vehicle (updated below)
    workload = np.zeros(len(primary_routes))  # demand in the primary route that must be filled by each vehicle (updated below)
    demand_filled = [0 for j in range(len(primary_routes))] # demand ultimately filled by each vehicle (updated below)
    realized_routes = []

    # Loop through vehicles
    for j in range(len(primary_routes)):
        if j == 0:
            workload[j] = primary_demands[j]
        else:
            workload[j] = max(0, primary_demands[j] - excess[j-1])

        if workload[j] == 0:
            excess[j] = excess[j-1] - primary_demands[j]
            demand_filled[j] = 0
        else:
            excess[j] = min(capacity * np.ceil(float(workload[j]) / capacity) - workload[j], overlap_demands[j])
            demand_filled[j] = workload[j] + excess[j]
        
        remaining_surplus = 0
        if j == 0:
            first[j] = primary_routes[0][0]
        else:
            remaining_surplus = excess[j-1]
        i = 0
        while remaining_surplus > 0:
            remaining_surplus -= inst.demands[primary_routes[j][i]]
            if remaining_surplus < 0:
                first[j] = primary_routes[j][i]
            elif remaining_surplus == 0 and i < len(primary_routes[j]) - 1:
                first[j] = primary_routes[j][i+1]
            elif i == len(primary_routes[j]) - 1:
                first[j] = 0
                last[j] = 0
                break
            i += 1    
        
        if j < len(primary_routes) - 1:
            remaining_surplus = excess[j]
            i = 0
            while remaining_surplus > 0:
                # fill demand of next shared customer
                # override default first and last customer if appropriate
                remaining_surplus -= inst.demands[overlapped_segments[j][i]]
                 
                if remaining_surplus <= 0:
                    # set last customer
                    last[j] = overlapped_segments[j][i]
                elif remaining_surplus > 0 and i == len(overlapped_segments[j]) - 1:
                    last[j] = overlapped_segments[j][i]
                    break
                i += 1
    
    # Determine realized routes based on updated first and last customers
    realized_routes = []
    for j in range(len(primary_routes)):

        # Create vehicle route
        if first[j] == 0:
            route = []  # vehicle doesn't leave depot
        else:
            first_index = inst.tour.index(first[j])
            last_index = inst.tour.index(last[j])
            route = inst.tour[first_index:last_index + 1]

        # Append to realized routes
        realized_routes.append(route)

    # Create full trips (i.e., segments) from realized routes
    segments = create_full_trips(inst, realized_routes, capacity, demand_filled)

    return segments


def implement_k_overlapped_alg_closed(demand_instance, primary_routes, extended_routes, capacity, route_size, overlap_size):
    """rotational k-overlapped strategy implemetation for flexible routing (inspired by closed chains)"""

    # Get customer instance
    inst = demand_instance
    tour = inst.tour
    # Set current tour and cumulative cost over all demand instances as best so far
    # Note: cumulative cost yields same tour ranking as average cost across demand instances
    best_segments = implement_k_overlapped_alg(inst, primary_routes, extended_routes, capacity, route_size, overlap_size)
    lowest_cumul_cost = sum(get_total_cost(inst, seg) for seg in best_segments)
    #print('Initial Routes: {}, Cost: {}'.format(best_routes, lowest_cumul_cost.round(2)))

    # Copy of tour (for rotating below)
    
    primary_routes_to_rotate = primary_routes
    extended_routes_to_rotate = extended_routes

    # Loop over all vehicles
    for c in range(len(primary_routes)-1):

        # Rotate primary and extended routes
        tour = tour[0:1] + tour[route_size+1:] + tour[1:route_size+1]
        inst.update_tour(tour)
        primary_routes_to_rotate = get_primary_routes(inst, route_size)
        extended_routes_to_rotate = get_extended_routes(inst, route_size, overlap_size)
        tour_cost = 0
        
        # Get cumulative cost over all demand instances
        segments = implement_k_overlapped_alg(inst, primary_routes_to_rotate, extended_routes_to_rotate, capacity, route_size,
                                                overlap_size)
        for seg in segments:
            tour_cost += get_total_cost(inst, seg)
        #print('Candidate Routes: {}, Cost: {}'.format(extended_routes_to_rotate, tour_cost.round(2)))
        if tour_cost < lowest_cumul_cost:
            # Set as new best routes and cost
            best_segments = segments
            lowest_cumul_cost = tour_cost
            #print('--> NEW BEST ROUTES: {}, COST: {}'.format(extended_routes_to_rotate, lowest_cumul_cost.round(2)))
    
    return best_segments

#---------------------------------------------------------------------------------

def create_instances(scenario, num_cust, cust_sims, dem_sims):
    """Returns cust_sims by dem_sims array of Instances"""

    np.random.seed(1)

    def gen_new_instance(num_cust, scenario):
        # Randomly generate depot locations
        depot_x = field_width * np.random.random(1)[0] # Depot x-coordinate
        depot_y = field_height * np.random.random(1)[0] # Depot y-coordinate

        # Generate customer locations
        new_xlocs = field_width * np.random.random(num_cust)  # x coordinates of all customers
        new_ylocs = field_height * np.random.random(num_cust)  # y coordinates of all customers

        # Generate demands depending on scenario
        if scenario == 'stochastic_customers':
            # Equal probability of selecting 0 or 8
            new_dems = list(np.random.choice([0,8], num_cust))
        elif scenario == 'binomial':
            # Binomial(8,0.5) distribution
            new_dems = list(np.random.binomial(8, 0.5, num_cust))
        else:
            # Uniformly distributed between 0 and 8
            new_dems = list(np.random.randint(0, 8, num_cust))


        # Return new instance
        new_xlocs = list(np.append([depot_x], new_xlocs))  # include depot in customer x-coords
        new_ylocs = list(np.append([depot_y], new_ylocs))  # include depot in customer y-coords
        new_dems = list(np.append([0], new_dems))  # include depot in customer demands
        return Instance(new_xlocs, new_ylocs, new_dems)

    def update_demands(inst, scenario):
        # Creates copy of instance with updated demands depending on scenario
        if scenario in ['baseline']:
            new_dems = list(np.random.randint(0, 8, num_cust))  # Uniformly distributed between 0 and 8
        else:
            raise ValueError('Scenario is undefined. Use \"baseline\" or define a new scenario in the update_demands function.')
            
        new_dems = list(np.append([0], new_dems))  # include depot in customer demands
        new_inst = Instance(inst.xlocs, inst.ylocs, new_dems, solve_TSP=False)
        new_inst.tour = inst.tour
        return new_inst

    # Create instance array with new customer instances
    instances = [[None for j in range(dem_sims)] for i in range(cust_sims)]
    customer_instances = [gen_new_instance(num_cust, scenario) for i in range(cust_sims)]

    # Create demand instances for each customer instance
    for i in range(cust_sims):
        for j in range(dem_sims):
            instances[i][j] = update_demands(customer_instances[i], scenario)

    return instances


# ---------------------------------------------------------------------------------

def set_best_tours(demand_instances, primary_routes, extended_routes, capacity, route_size, overlap_size):
    """Updates the tour of all instances to the sequence that minimizes the average cost of the routes over all demand instances.
    Assumes all instances in list demand_instances have identical customer locations."""

    # Get any customer instance
    inst = demand_instances[0]
    # Set current tour and cumulative cost over all demand instances as best so far
    # Note: cumulative cost yields same tour ranking as average cost across demand instances
    best_tour = inst.tour
    segments = implement_k_overlapped_alg(inst, primary_routes, extended_routes, capacity, route_size, overlap_size)
    lowest_cumul_cost = sum([get_total_cost(inst, seg) for seg in segments for inst in demand_instances])

    # Copy of tour (for rotating below)
    tour = inst.tour

    # Loop over all customers
    for c in range(inst.size):

        # Rotate tour by one customer (keeps depot at very first spot)
        tour = tour[0:1] + tour[2:] + tour[1:2]
        inst.update_tour(tour)
        tour_cost = 0

        # Get cumulative cost over all demand instances
        for inst in demand_instances:
            segments = implement_k_overlapped_alg(inst, primary_routes, extended_routes, capacity, route_size,
                                                  overlap_size)
            for seg in segments:
                tour_cost += get_total_cost(inst, seg)

        # Set new best tour if lower cost
        if tour_cost < lowest_cumul_cost:
            best_tour = tour
            lowest_cumul_cost = tour_cost

    # Update tour for all demand instances in this customer row
    for inst in demand_instances:
        inst.update_tour(best_tour)
    return



