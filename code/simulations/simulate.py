### simulate.py
### this file runs the simulations

route_size = 5  # Number of customers in fixed primary route
N_values = [5 * m for m in [1]]
# N_values = [5*m for m in [1,2,4,8,16]]
# N_values = [80]
dem_sims = 200  # Number of demand instances per problem size
cust_sims = 500  # Number of customer instances per problem size
d_max = 8  # Maximum customer demand
d_min = 0  # Minimum customer demand
Q = 20  # Vehicle capacity
area_width = 100
area_height = 100

# Single centrally located depot
depots = dict([("Depot_1", Depot(ID="Depot_1", x=50, y=50))])
my_depot = depots['Depot_1']

#-------------------------------------------------------------------------

def run_rotating():

    ##### Initialize arrays for results #####

    lowerbound_sims = np.zeros((len(N_values), cust_sims, dem_sims))
    primary_sims = np.zeros((len(N_values), cust_sims, dem_sims, 4))  # For circular, radial, total, and trip count
    overlapped_sims = np.zeros((len(N_values), cust_sims, dem_sims, 4))
    full_flex_sims = np.zeros((len(N_values), cust_sims, dem_sims, 4))
    SDVRP_sims = np.zeros((len(N_values), cust_sims, dem_sims, 2))  # For total cost and trip count

    ##### Run simulation #####

    for n in range(len(N_values)):
        print('Solving problems of size %s...' % N_values[n], flush=True)

        # Create customers
        random.seed(10)
        for i in range(cust_sims):
            random.seed(1000 * N_values[n] * (i + 1))
            print('Starting customer instance', i + 1, '...', flush=True)

            # Generate set of N customers
            customers = dict([('Cust_' + str(c + 1),
                               Customer(ID='Cust_' + str(c + 1),
                                        x=random.uniform(0, area_width),
                                        y=random.uniform(0, area_height),
                                        d=random.choice([d_min, d_max])))
                              # d = random.randint(d_min,d_max)))
                              for c in range(N_values[n])])

            # Get tour
            obj, X, runtime = solve_Concorde_TSP(customers)
            tau = get_best_tour(X, customers, my_depot, Q, route_size, d_min, d_max, dem_sims)

            # Demand simulations
            random.seed(0)
            for j in range(dem_sims):

                # Update customer demands
                for cust in customers:
                    # customers[cust].d = random.randint(d_min,d_max)
                    customers[cust].d = random.choice([d_min, d_max])

                # Solve network models
                lowerbound_sims[n, i, j] = get_lowerbound(customers, my_depot, Q)
                primary_sims[n, i, j, :] = solve_primary(tau, customers, my_depot, Q, route_size)
                overlapped_sims[n, i, j, :] = solve_overlapped(tau, customers, my_depot, Q, route_size)
                full_flex_sims[n, i, j, :] = solve_fully_flexible(tau, customers, my_depot, Q)

                # SDVRP route costs
                demand_points = get_demand_points(customers)
                obj, runtime, num_trips = solve_OR_Tools_VRP(demand_points, depots, Q)
                SDVRP_sims[n, i, j, :] = (obj, num_trips)
                # print('SDVRP completed in %s seconds' %np.round(runtime,2), flush=True)

    print('Complete.')


#-------------------------------------------------------------------------

def run_non_rotating():
    ##### Initialize arrays for results #####

    lowerbound_sims = np.zeros((len(N_values), cust_sims, dem_sims))
    primary_sims = np.zeros((len(N_values), cust_sims, dem_sims, 4))  # For circular, radial, total, and trip count
    overlapped_sims = np.zeros((len(N_values), cust_sims, dem_sims, 4))
    full_flex_sims = np.zeros((len(N_values), cust_sims, dem_sims, 4))
    SDVRP_sims = np.zeros((len(N_values), cust_sims, dem_sims, 2))  # For total cost and trip count

    ##### Run simulation #####

    for n in range(len(N_values)):
        print('Solving problems of size %s...' % N_values[n], flush=True)

        # Create customers
        for i in range(cust_sims):
            random.seed(1000 * N_values[n] * (i + 1))
            print('Starting customer instance', i + 1, '...', flush=True)

            # Generate set of N customers
            customers = dict([('Cust_' + str(c + 1),
                               Customer(ID='Cust_' + str(c + 1),
                                        x=random.uniform(0, area_width),
                                        y=random.uniform(0, area_height),
                                        d=random.choice([d_min, d_max])))
                              # d = random.randint(d_min,d_max)))
                              for c in range(N_values[n])])

            # Get tour
            obj, X, runtime = solve_Concorde_TSP(customers)
            tau = get_sequence(customers, X)

            # Demand simulations
            random.seed(0)
            for j in range(dem_sims):

                # Update customer demands
                for cust in customers:
                    # customers[cust].d = random.randint(d_min,d_max)
                    customers[cust].d = random.choice([d_min, d_max])

                # Solve network models
                lowerbound_sims[n, i, j] = get_lowerbound(customers, my_depot, Q)
                primary_sims[n, i, j, :] = solve_primary(tau, customers, my_depot, Q, route_size)
                overlapped_sims[n, i, j, :] = solve_overlapped(tau, customers, my_depot, Q, route_size)
                full_flex_sims[n, i, j, :] = solve_fully_flexible(tau, customers, my_depot, Q)

                # SDVRP route costs
                demand_points = get_demand_points(customers)
                obj, runtime, num_trips = solve_OR_Tools_VRP(demand_points, depots, Q)
                SDVRP_sims[n, i, j, :] = (obj, num_trips)
                # print('SDVRP completed in %s seconds' %np.round(runtime,2), flush=True)

    print('Complete.')


def export_results():
    primary = pd.DataFrame(np.mean(primary_sims, axis=(1,2)), columns=['Circular','Radial','Total','Trips'])
    overlapped = pd.DataFrame(np.mean(overlapped_sims, axis=(1,2)), columns=['Circular', 'Radial','Total','Trips'])
    full_flex = pd.DataFrame(np.mean(full_flex_sims, axis=(1,2)), columns=['Circular', 'Radial','Total','Trips'])
    lowerbound = pd.DataFrame(np.mean(lowerbound_sims, axis=(1,2)), columns=['Total'])
    SDVRP = pd.DataFrame(np.mean(SDVRP_sims, axis=(1,2)), columns=['Total','Trips'])

    combined = pd.concat([primary,overlapped,full_flex,SDVRP,lowerbound], axis=1,
                         keys = ['Primary','Overlapped','Full Flexibility','Reoptimization','Lowerbound'])
    combined['N'] = N_values
    combined['M'] = np.divide(N_values,route_size).astype(int)
    combined.to_excel('output/test/Comb_stochCust_M1.xlsx')
    combined.round(3)