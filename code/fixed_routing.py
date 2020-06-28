### fixed_routing.py
### this file contains the fixed routing algortihms and supporting methods

#-------------------------------------------------------------------------

def get_breakpoint_indices(demands, Q):
    """
    Takes in list of demands and an integer capacity, and returns a list of indices based on the demand list
    where the truck will detour to the depot to refill and then return to the route.
    """

    indices = []
    k = 1  # Scalar for capacity to track number of refill trips
    agg = 0  # Tracks aggregate route demand filled

    for i in range(0, len(demands)):

        rem_demand = demands[i]
        while rem_demand > 0:

            if agg < k * Q and k * Q < agg + demands[i]:
                indices.append(i)  # breakpoint
                indices.append(i)  # returnpoint
                k += 1
            elif agg < k * Q and k * Q == agg + demands[i]:
                if i < len(demands) - 1:
                    indices.append(i)  # breakpoint
                    indices.append(i + 1)  # returnpoint
                k += 1
            elif agg == (k - 1) * Q and k * Q == agg + demands[i]:
                if i < len(demands) - 1:
                    indices.append(i)  # breakpoint
                    indices.append(i + 1)  # returnpoint
                k += 1
            else:
                pass

            rem_demand -= Q

        agg += demands[i]

    return indices


#-------------------------------------------------------------------------

def get_primary_routes(customers, tau, route_size):
    """
    Creates primary fixed routes. Receives as input a dictionary of customer objects, a sequence (dictionary) of customer objects,
    and the number (int) of customers to include in each route. Outputs a dictionary of route objects with route number as the key
    and the object as the value.
    """

    M = int(len(tau) / route_size)  # Number of routes
    routes = {}

    for r in range(M):
        sub_seq = dict([(i + 1, tau[i + 1]) for i in range(r * route_size, (r + 1) * route_size)])
        routes[r + 1] = Route(sub_seq, r + 1)
    return routes


#-------------------------------------------------------------------------

def get_start_position(sub_tau, capacity):
    """
    Determines the position of the customer whose demand cannot be completely filled. Receives a dictionary with sequence of customer objects and
    an integer representing total demand allowed to be filled. Returns a tuple with (1) the position of the customer whose demand cannot be completely filled
    and (2) the amount of demand for the next truck to fill at the customer. Returns (0,0) if the truck doesn't need to leave the depot.
    """
    positions = list(sub_tau.keys())

    demand = 0  # Demand counter
    j = 0  # Key tracker
    i = 0  # Index counter

    # Fill customers' demand until capacity exhausted
    while demand <= capacity:

        if i < len(positions):

            j = positions[i]
            demand += sub_tau[j].d  # Add in demand of customer number i
            i += 1

        else:
            return (0, 0)

    return (positions[i - 1], demand - capacity)


#-------------------------------------------------------------------------

def solve_fully_flexible(tau, customers, depot, Q):
    """
    Determines the cost of vehicle routing and the number of trips needed with fully flexible routes.

    INPUTS
    tau : dictionary with customer sequence
    customers : dictionary of customer objects
    depot : SINGLE depot object
    Q : truck capacity (int)

    OUTPUTS
    Tuple with (1) circular cost, (2) radial cost, (3) combined cost, and (4) number of trips needed to meet all demand.
    """
    # Get pairwise distances
    nodes = {}
    nodes.update(depots)
    nodes.update(customers)
    distances = calc_dist_matrix(nodes)

    # Keep customers with non-zero demand
    nonzero_tau = dict([(i, tau[i]) for i in tau if tau[i].d > 0])

    # Use bin-packing to determine  route endpoints (list indices)
    demands = [nonzero_tau[i].d for i in nonzero_tau]
    if len(demands) > 0:
        endpoints = [0] + get_breakpoint_indices(demands, Q) + [len(demands) - 1]
        assert len(endpoints) % 2 == 0, "Assertion Error: There is not an even number of endpoints."
        num_trips = len(endpoints) / 2
    else:
        return (0, 0, 0, 0)  # No demands to fill, so no transportation costs or trips

    # Use end points to create Routes and calculate costs
    greedy_routes = {}
    circular_costs = []
    radial_costs = []
    total_costs = []

    j = 1
    for e in range(0, len(endpoints) - 1, 2):
        # Create new route
        start = endpoints[e]  # Position of first customer
        end = endpoints[e + 1]  # Position of last customer
        sub_seq = dict(
            [(i, nonzero_tau[i]) for i in nonzero_tau if i >= [*nonzero_tau][start] and i <= [*nonzero_tau][end]])
        greedy_routes[j] = Route(sub_seq, j)

        # TO-DO: Need to update to accomodate split demand

        # Calculate radial costs at beginning and end of route
        radial = distances[depot.ID, greedy_routes[j].customers[0].ID] + distances[
            depot.ID, greedy_routes[j].customers[-1].ID]
        radial_costs.append(radial)

        # Calculate circular cost
        route_edges = get_edge_list(greedy_routes[j], depot)  # Edge list
        circular = sum([distances[e] for e in route_edges])
        circular -= radial  # TEMP FIX: remove radial cost from circular to prevent double counting
        circular_costs.append(circular)

        # Total cost
        total_costs.append(circular + radial)
        j += 1

    """
    ### Generate report ###

    print('N=%s, M=%s, Q=%s \n' %(len(customers),len(greedy_routes),Q))

    for j in greedy_routes:
        print("Truck %s Route (pos, d) -->" %j, [(list(greedy_routes[j].sequence.keys())[i],
                                                 greedy_routes[j].customers[i].ID,
                                                 greedy_routes[j].customers[i].d) for i in range(greedy_routes[j].n)])
    print('')
    print('Starting Position (s), Workload (w), Excess Capacity (e), Final Position (f):')
    for j in greedy_routes:
        print('Route %s: s=%s, w=%s, e=%s, f=%s'
              %(j,[*greedy_routes[j].sequence][0],greedy_routes[j].d, Q-greedy_routes[j].d, [*greedy_routes[j].sequence][-1]))
    print('')
    print('Circular Costs --> ', np.round(circular_costs))
    print('Radial Costs --> ', np.round(radial_costs))
    print('Total Costs --> ', np.round(np.add(radial_costs,circular_costs)))

    """

    return sum(circular_costs), sum(radial_costs), sum(total_costs), num_trips


#-------------------------------------------------------------------------

def solve_primary(tau, customers, depot, Q, route_size):
    """
    Determines the cost of vehicle routing and the number of trips needed with fixed non-overlapping routes. Assumes M = N/route_size >= 1.

    INPUTS
    tau : dictionary with customer sequence
    customers : dictionary of customer objects
    depot : SINGLE depot object
    Q : truck capacity (int)
    route_size: number of customers per route (int)

    OUTPUTS
    Tuple with (1) circular cost, (2) radial cost, (3) combined cost, and (4) number of trips needed to meet all demand.
    """

    N = len(customers)  # Number of customers
    M = int(N / route_size)  # Number of trucks (assume N divisible by route_size)

    # Create set of nodes with both depots and customers
    nodes = {}
    nodes.update(depots)
    nodes.update(customers)

    # Calculate pairwise distance matrix
    distances = calc_dist_matrix(nodes)

    # Create primary routes
    primary_routes = get_primary_routes(customers, tau, route_size)

    # Solve each primary route as if fully flexible
    circular_costs = []
    radial_costs = []
    total_costs = []
    num_trips = []
    for j in primary_routes:
        circular, radial, total, trips = solve_fully_flexible(primary_routes[j].sequence, customers, depot, Q)
        circular_costs.append(circular)
        radial_costs.append(radial)
        total_costs.append(total)
        num_trips.append(trips)

    return sum(circular_costs), sum(radial_costs), sum(total_costs), sum(num_trips)


#-------------------------------------------------------------------------

def solve_overlapped(tau, customers, depot, Q, route_size, k=None):
    """
    Determines the cost of vehicle routing and the number of trips needed with a priori ADJACENT overlapping routes.
    Assumes M = N/route_size >=1.

    INPUTS
    tau : dictionary with customer sequence
    customers : dictionary of customer objects
    depot : SINGLE depot object
    Q : truck capacity (int)
    route_size: number of customers per route (int)

    OUTPUTS
    Tuple with (1) circular cost, (2) radial cost, (3) combined cost, and (4) number of trips needed to meet all demand.
    """

    N = len(customers)  # Number of customers
    M = int(N / route_size)  # Number of trucks (assume N divisible by route_size)
    num_trips = 0

    # Create set of nodes with both depots and customers
    nodes = {}
    nodes.update(depots)
    nodes.update(customers)

    # Calculate pairwise distance matrix
    distances = calc_dist_matrix(nodes)

    # Create fixed routes
    primary_routes = get_primary_routes(customers, tau, route_size)

    # Keep just non-zero demand customers in primary routes
    nonzero_routes = {}
    for j in primary_routes:
        my_route = primary_routes[j]
        sub_seq = dict([(i, my_route.sequence[i])
                        for i in my_route.sequence if my_route.sequence[i].d > 0])
        nonzero_routes[j] = Route(sub_seq, j)

    slist = []  # Position of truck's starting customer
    wlist = []  # Truck workload for primary customers
    elist = []  # Truck's surplus capacity upon finishing primary route
    dlist = []  # The amount of demand for customer s_j that truck j serves
    flist = []  # Final customer for each truck

    ### Initiate algorithm with Truck 1 ###
    j = 1  # Route key
    my_route = nonzero_routes[j]

    if my_route.n == 0:
        slist.append(0)
        dlist.append(0)
        wlist.append(0)
        elist.append(0)
    else:
        s = min(i for i in tau if tau[i].d > 0)  # First customer with non-zero demand
        d = tau[s].d
        w = my_route.d
        slist.append(s)
        dlist.append(d)
        wlist.append(w)
        e = np.ceil(wlist[0] / Q) * Q - wlist[0]
        elist.append(e)

    ###  Calculate route features for Truck 2 ###

    if M >= 2:
        j = 2  # Route key
        my_route = nonzero_routes[j]

        if my_route.n == 0:
            slist.append(0)
            dlist.append(0)
            wlist.append(0)
            elist.append(0)
        else:
            s, d = get_start_position(my_route.sequence, elist[0])
            slist.append(s)
            dlist.append(d)
            w = max(0, my_route.d - elist[j - 2])
            wlist.append(w)
            e = np.ceil(wlist[j - 1] / Q) * Q - wlist[j - 1]
            elist.append(e)

        # Determine Truck 1's final customer
        if slist[0] == 0:  # Truck 1 doesn't leave depot
            flist.append(0)
        elif my_route.sequence == {}:  # Truck 2 has no customers so truck 1 covers only its own customerss
            f = max(i for i in nonzero_routes[1].sequence if tau[i].d > 0)
            flist.append(f)
        elif slist[
            1] == 0:  # Truck 2 has customers but doesn't leave the depot so truck 1 covers all of truck 2's customers
            f = max(i for i in my_route.sequence if tau[i].d > 0)
            flist.append(f)
        elif d == tau[s].d:  # Truck 2 covers all demand of customer s
            f = max(i for i in tau if i < s if tau[i].d > 0)  # Truck 1 ends with the preceding non-zero customer
            flist.append(f)
        else:  # Trucks 1 and 2 split customer s
            flist.append(slist[1])

            ### Calculate route features for Trucks 3 through M ###
    if M >= 3:

        for j in range(3, M + 1):  # Truck number, NOT index

            my_route = nonzero_routes[j]

            # Starting customer
            if slist[j - 2] == j * route_size + 1:
                s = j * route_size + 1
                slist.append(s)
            else:
                s, d = get_start_position(my_route.sequence, elist[j - 2])
                slist.append(s)
                dlist.append(d)
            # Remaining workload for nonzero route
            w = max(0, my_route.d - elist[j - 2])
            wlist.append(w)

            # Excess capacity after primary route
            e = np.ceil(wlist[j - 1] / Q) * Q - wlist[j - 1]
            elist.append(e)

            # Final customer for PREVIOUS truck j-1

            if slist[j - 2] == 0:  # Truck j-1 doesn't leave the depot
                flist.append(0)  # No final customer for Truck j-1

            elif my_route.sequence == {}:  # Truck j has no customers so truck j-1 covers only its own customers
                f = max(i for i in nonzero_routes[j - 1].sequence if tau[i].d > 0)
                flist.append(f)

            elif slist[
                j - 1] == 0:  # Truck j has customers but doesn't leave the depot so truck 1 covers all of truck 2's customers
                f = max(i for i in my_route.sequence if tau[i].d > 0)
                flist.append(f)

            elif d == tau[s].d:  # Truck j covers all demand of customer s
                f = max(i for i in tau if i < s if tau[i].d > 0)  # Truck j-1 ends with the preceding non-zero customer
                flist.append(f)

            else:  # Trucks j-1 and j split customer s
                flist.append(slist[j - 1])  # Truck j-1 finishes exactly where Truck j starts

    # Final customer for route M
    if slist[M - 1] == 0:
        flist.append(0)  # Never left depot
    else:
        f = max(i for i in tau if tau[i].d > 0)
        flist.append(f)

    ### Construct REALIZED routes based on demands ###
    realized_routes = {}
    for j in range(1, M + 1):  # truck number, NOT index

        # Create route with non-zero customers between start and finish points
        positions = list(range(slist[j - 1], flist[j - 1] + 1))
        sub_seq = dict([(i, tau[i]) for i in positions if i != 0 and tau[i].d > 0])
        realized_routes[j] = Route(sub_seq, j)

        # Set all customers split demand to the full demand (default)
        for cust in realized_routes[j].customers:
            cust.dsplit[j] = cust.d
        if realized_routes[j].customers != []:  # (Otherwise truck doesn't leave the depot)
            # Check for splitting in route's first customer
            first_cust = realized_routes[j].sequence[slist[j - 1]]
            if first_cust.d != dlist[j - 1]:
                # Demand is split
                first_cust.dsplit[j - 1] = first_cust.d - dlist[j - 1]  # Add entry
                first_cust.dsplit[j] = dlist[j - 1]  # Overwrite entry
                # Update route demands
                realized_routes[j - 1].d = realized_routes[j - 1].d - dlist[j - 1]
                realized_routes[j].d = realized_routes[j].d - first_cust.d + dlist[j - 1]

    ### Get transportation costs ###

    circular_costs = []
    radial_costs = []
    total_costs = []
    # Get edge list for each realized route and calculate circular distance
    for j in range(1, M + 1):  # truck number, NOT index

        my_route = realized_routes[j]

        if my_route.n == 0:
            circular_costs.append(0)
            radial_costs.append(0)
            total_costs.append(0)

        else:

            # Refill costs
            split_demand_list = [cust.dsplit[j] for cust in my_route.customers]  # This is the split demand list
            detour_points = get_breakpoint_indices(split_demand_list, Q)
            customer_list = [my_route.customers[p].ID for p in detour_points]
            radial = sum([distances[depot.ID, customers[i].ID] for i in customer_list])
            # Update trip counts for refill trips
            num_trips += len(detour_points) / 2 + 1

            # Circular cost
            route_edges = get_edge_list(my_route, depot)  # Edge list
            for i in range(0, len(detour_points), 2):
                if detour_points[i] != detour_points[i + 1]:
                    route_edges.remove(route_edges[detour_points[i] + 1])
            circular = sum([distances[e] for e in route_edges])

            # TEMP FIX: switch initial/final depot trip from circular cost to radial cost
            extrem_dists = distances[route_edges[0]] + distances[route_edges[len(route_edges) - 1]]
            circular -= extrem_dists
            radial += extrem_dists

            circular_costs.append(circular)
            radial_costs.append(radial)
            total_costs.append(circular + radial)

    ### Generate report ###
    """
    print('N=%s, M=%s, Q=%s \n' %(len(customers),M,Q))

    for j in primary_routes:
        print("Truck %s Route (pos, d) -->" %j, [(list(primary_routes[j].sequence.keys())[i],
                                                  primary_routes[j].customers[i].ID,
                                                  primary_routes[j].customers[i].d) for i in range(primary_routes[j].n)])
    print('')
    for j in nonzero_routes:
        print("Truck %s Non-Zero Route (pos, d) -->" %j, [(list(nonzero_routes[j].sequence.keys())[i],
                                                           nonzero_routes[j].customers[i].ID,
                                                           nonzero_routes[j].customers[i].d) for i in range(nonzero_routes[j].n)])
    print('')
    print('Starting Position (s), Workload (w), Excess Capacity (e), Final Position (f):')
    for j in range(M):
        print('Route %s: s=%s, w=%s, e=%s, f=%s' %(j+1,slist[j],wlist[j],elist[j],flist[j]))

    print('')
    for j in realized_routes:
        print("Truck %s Realized Route (pos) -->" %j, [(list(realized_routes[j].sequence.keys())[i],
                                                        realized_routes[j].customers[i].ID) for i in range(realized_routes[j].n)])
        print('Actual Demand Filled:', realized_routes[j].d)

    print('')
    print('Circular Costs --> ', np.round(circular_costs))
    print('Radial Costs --> ', np.round(radial_costs))
    print('Total Costs --> ', np.round(np.add(radial_costs, circular_costs)))
    """
    return sum(circular_costs), sum(radial_costs), sum(total_costs), num_trips

