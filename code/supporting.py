### supporting.py
### this file includes supporting methods and utilities

#-------------------------------------------------------------------------

def calc_dist_matrix(nodes):
    """
    Compute pairwise distance matrix. Takes in dictionary of objects with locations (e.g. customers and depots).
    Returns dictionary of Euclidian distances between all possible node pairs. TO DO: Update function name to dict.
    """

    return dict(
        [((i, j), math.sqrt((nodes[i].x - nodes[j].x) ** 2
                            + (nodes[i].y - nodes[j].y) ** 2))
         for i in nodes
         for j in nodes]
    )


#-------------------------------------------------------------------------

def get_edge_list(route, depot):
    """
    Creates list of Route edges. Receives Route object and Depot object as inputs and returns a list of tuples of NODE IDs
    representing edges between nodes in a route.
    """

    # List of node IDs (positions)
    positions = [cust.ID for cust in route.customers]

    # Replace position of 0 with depot ID
    for i in range(len(positions)):
        if positions[i] == 0:
            positions[i] = depot.ID

            # First edge is between depot and first customer
    route_edges = [(depot.ID, positions[0])]

    # Create edges between customers
    for i in range(route.n - 1):
        route_edges.append((positions[i], positions[i + 1]))

    # Create final edge to depot
    route_edges.append((positions[route.n - 1], depot.ID))

    return route_edges


#-------------------------------------------------------------------------

def get_sequence(customers, X):
    """
    Creates a customer seqence by linking together selected edges. Takes in dictionary of customer objects and dictionary of
    routing edge decisions (i.e., (i,j) customer ID tuples and 0-1 decisions for routing). Outputs a dictionary with (position in
    sequence, customer object) as the key-value pairs.
    """

    # Select first customer object to initiate path
    i = list(customers.keys())[0]
    sequence = [customers[i]]

    # Record that you have visited one customer already
    n_done = 1

    # While you have not yet visited all customers...
    while n_done <= len(customers):

        # Find next customer along the given path
        for j in customers:
            if ((i, j) in X) and (round(X[(i, j)]) == 1):
                sequence.append(customers[j])  # Add customer to sequence
                i = j  # Set the last visited customer to this customer
                n_done += 1  # Increase the number of customers visited
                break  # Break this for loop since you found the model's selected customer

    return dict([(n + 1, sequence[n]) for n in range(len(sequence) - 1)])


#-------------------------------------------------------------------------

def get_routing_solution(tau):
    """Given a customer sequence (dictionary of position-customer pairs), returns the equivalent dictionary of routing decision variables."""
    new_X = dict([((tau[i].ID, tau[j].ID), 0) for i in tau for j in tau])
    for i in range(len(tau) - 1):
        start_cust = tau[list(tau.keys())[i]]
        next_cust = tau[list(tau.keys())[i + 1]]
        new_X[(start_cust.ID, next_cust.ID)] = 1
    # Add final closing arc
    new_X[(next_cust.ID, tau[list(tau.keys())[0]].ID)] = 1
    return new_X


def get_demand_points(customers):
    """Given a dictionary of customer objects, returns a dictionary of equivalent customers with binary demands."""
    demand_points = {}
    for c in customers:
        for i in range(int(customers[c].d)):
            demand_points[customers[c].ID + '_' + str(i + 1)] = Customer(ID=customers[c].ID + '_' + str(i + 1),
                                                                         x=customers[c].x,
                                                                         y=customers[c].y,
                                                                         d=1)
    return demand_points


#-------------------------------------------------------------------------

def get_lowerbound(customers, depot, Q):
    """Returns a lowerbound on transportation cost given a dictionary of customer objects,
    a single depot object, and truck capacity Q."""

    # Get pairwise distances
    nodes = {}
    nodes.update(depots)
    nodes.update(customers)
    distances = calc_dist_matrix(nodes)

    # Return lower bound
    return (2 / Q) * sum([customers[i].d * distances[(depot.ID, customers[i].ID)] for i in customers])


#-------------------------------------------------------------------------

def get_best_tour(X, custs, depot, Q, route_size, d_min, d_max, dem_sims):
    cumul_costs = np.zeros(len(custs))
    taulist = []

    # Get sequence from intial TSP solution
    first_tau = get_sequence(custs, X)
    taulist.append(first_tau)

    # Get cumulative cost across all demand instances for first rotation
    random.seed(0)
    for j in range(dem_sims):
        for cust in custs:
            # custs[cust].d = random.randint(d_min,d_max)
            custs[cust].d = random.choice([d_min, d_max])

        cumul_costs[0] += solve_overlapped(first_tau, custs, depot, Q, route_size)[2]

    # Rotate and find cumulative costs across all demand sims
    rotated_tau = first_tau

    for r in range(1, len(custs)):
        rotated_customers = list(rotated_tau.values())[1:] + [list(rotated_tau.values())[0]]
        rotated_tau = dict([(c, rotated_customers[i]) for c, i in zip(first_tau, range(len(first_tau)))])
        taulist.append(rotated_tau)
        random.seed(0)
        for j in range(dem_sims):
            for cust in custs:
                # custs[cust].d = random.randint(d_min,d_max)
                custs[cust].d = random.choice([d_min, d_max])
            cumul_costs[r] += solve_overlapped(rotated_tau, custs, depot, Q, route_size)[2]

    best_r = np.argmin(cumul_costs)
    return taulist[best_r]