import pandas as pd
import time
from supporting import *
from copy import deepcopy

# GLOBAL VARIABLES
field_width = 100 # Customer location has x-coordinate in (0, field_width)
field_height = 100 # Customer location has y-coordinate in (0, field_height)
#depot_x = 50 # Depot x-coordinate
#depot_y = 50 # Depot y-coordinate


#---------------------------------------------------------------------------------

def create_report(inst, scenario, strategy, segments):
    """Gets costs and c reates new entry for simulation results"""
    trips = [scenario, inst.size, strategy, 'trip count', get_trip_count(segments)]
    radial = [scenario, inst.size, strategy, 'radial cost', sum([get_radial_cost(inst, seg) for seg in segments])]
    circular = [scenario, inst.size, strategy, 'circular cost', sum([get_circular_cost(inst, seg) for seg in segments])]
    total = [scenario, inst.size, strategy, 'total cost', sum([get_total_cost(inst, seg) for seg in segments])]
    return pd.DataFrame(data=[trips, radial, circular, total],
                        columns=['Scenario', 'Number of Customers', 'Routing Strategy', 'Metric', 'Value'])


#---------------------------------------------------------------------------------

def simulate(scenario, problem_sizes, capacity, route_size, overlap_size, cust_sims, dem_sims):

    # Start timers
    start = time.time()
    pt, dt, ot, ct, ft, fct, rt, st = 0, 0, 0, 0, 0, 0, 0, 0

    # Create timestamp for backup outputs
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")

    # Print simulation parameters
    print('--- SIMULATION PARAMETERS ---')
    print('Start time:', timestamp)
    print('Scenario Name:', scenario)
    print('Problem sizes:', problem_sizes)
    print('Vehicle capacity:', capacity)
    print('Primary route size:', route_size)
    print('Overlap size:', overlap_size)
    print('Customer instances:', cust_sims)
    print('Demand instances:', dem_sims)
    print()

    # Initialize arrays to store results
    sim_results = pd.DataFrame(columns=['Scenario', 'Number of Customers', 'Routing Strategy', 'Metric', 'Value'])

    # Loop through each problem size
    for num_cust in problem_sizes:

        print('Starting problems of size {}'.format(num_cust))

        new_pt = time.time()
        # Create all customer and demand instances for this problem size
        print('Creating customer instances')
        instances = create_instances(scenario, num_cust, cust_sims, dem_sims)

        # Find cost minimizing starting customer / tour sequence for each set of customer locations
        print('Finding best tour across demand sets')
        for row in instances:
            inst = row[0]  # customer instance
            primary_routes = get_primary_routes(inst, route_size)
            extended_routes = get_extended_routes(inst, route_size, overlap_size)
            # set average cost-minimizing tour
            set_best_tours(row, primary_routes, extended_routes, capacity, route_size, overlap_size)
        pt += time.time() - new_pt

        # Loop through instances and find instance cost for different strategies

        for i in range(cust_sims):
            for j in range(dem_sims):

                # Get instance from array
                inst_copy = deepcopy(instances[i][j])
                

                try:
                    # Solve dedicated routing
                    inst = deepcopy(inst_copy)
                    new_dt = time.time()
                    primary_routes = get_primary_routes(inst, route_size)
                    segments = create_full_trips(inst, primary_routes, capacity)
                    new_rows = create_report(inst, scenario, 'dedicated', segments)
                    sim_results = sim_results.append(new_rows, ignore_index=True)
                    dt += time.time() - new_dt

                    # Solve k-overlapped routing
                    inst = deepcopy(inst_copy)
                    new_ot = time.time()
                    primary_routes = get_primary_routes(inst, route_size)
                    extended_routes = get_extended_routes(inst, route_size, overlap_size)
                    segments = implement_k_overlapped_alg(inst, primary_routes, extended_routes, capacity, route_size, overlap_size)
                    new_rows = create_report(inst, scenario, 'overlapped', segments)
                    sim_results = sim_results.append(new_rows, ignore_index=True)
                    ot += time.time() - new_ot
                    
                    
                    # Solve full overlapped routing
                    inst = deepcopy(inst_copy)
                    new_ft = time.time()
                    segments = create_full_trips(inst, [inst.tour[1:]], capacity)
                    new_rows = create_report(inst, scenario, 'fully flexible', segments)
                    sim_results = sim_results.append(new_rows, ignore_index=True)
                    ft += time.time() - new_ft
                    
                    # Solve rotational k-overlapped routing (inspired by closed chains)
                    inst = deepcopy(inst_copy)
                    new_ct = time.time()
                    primary_routes = get_primary_routes(inst, route_size)
                    extended_routes = get_extended_routes(inst, route_size, overlap_size)
                    segments = implement_k_overlapped_alg_closed(inst, primary_routes, extended_routes, capacity, route_size, overlap_size)
                    new_rows = create_report(inst, scenario, 'overlapped closed', segments)
                    sim_results = sim_results.append(new_rows, ignore_index=True)
                    ct += time.time() - new_ct

                    
                    # Solve rotational full overlapped routing
                    inst = deepcopy(inst_copy)
                    new_fct = time.time()
                    primary_routes = get_primary_routes(inst, route_size)
                    extended_routes = get_extended_routes(inst, route_size, inst.size)
                    segments = implement_k_overlapped_alg_closed(inst, primary_routes, extended_routes, capacity, route_size, inst.size)
                    new_rows = create_report(inst, scenario, 'fully flexible closed', segments)
                    sim_results = sim_results.append(new_rows, ignore_index=True)
                    fct += time.time() - new_fct

                    # Solve reoptimization
                    inst = deepcopy(inst_copy)
                    new_rt = time.time()
                    segments = solve_SDVRP(inst, capacity)
                    new_rows = create_report(inst, scenario, 'reoptimization', segments)
                    sim_results = sim_results.append(new_rows, ignore_index = True)
                    rt += time.time() - new_rt

                except Exception as e:
                    print('ERROR: {}'.format(e))
                    print('WARNING: Simulation failed to complete. Printing info for last Instance and returning Instance object.')
                    print(inst.demands)
                    print(inst.xlocs)
                    print(inst.ylocs)
                    print(inst.tour)
                    return inst

            # Save backup of data
            new_st = time.time()
            sim_results.to_csv('temp/backup_{}.csv'.format(timestamp))
            st += time.time() - new_st

            end = time.time()
            print('Customer instance {} complete. Time elapsed: {:.2f} min'.format(i + 1, (end-start)/60))

        print('Problems of size {} complete'.format(num_cust))

    print('Simulation complete.')
    print()
    print('--- RUNTIME BREAKDOWN ---')
    print('Setup: {:.2f} min'.format(pt/60))
    print('Dedicated: {:.2f} min'.format(dt/60))
    print('Overlapped: {:.2f} min'.format(ot/60))
    print('Overlapped Closed: {:.2f} min'.format(ct/60))
    print('Full Flex.: {:.2f} min'.format(ft/60))
    print('Full Flex. Closed: {:.2f} min'.format(fct/60))
    print('Reoptimization: {:.2f} min'.format(rt/60))
    print('Saving: {:.2f} min'.format(st/60))

    return sim_results


#---------------------------------------------------------------------------------

if __name__ == "__main__":

    # --- Baseline simulation ---
    # Demand uniformly distributed in [0,8]
    # Route size: 5
    # Overlap size: 5
    results = simulate(scenario = 'baseline', problem_sizes = [5,10,20,40,80], capacity = 20, route_size = 5, overlap_size = 5, cust_sims = 30, dem_sims = 200)

    # --- Baseline k=3 simulation ---
    # Demand uniformly distributed in [0,8]
    # Route size: 5
    # Overlap size: 3
    #results = simulate(scenario = 'baseline_k3', problem_sizes = [5,10,20,40,80], capacity = 20, route_size = 5, overlap_size = 3, cust_sims = 30, dem_sims = 200)

    # --- Baseline k=1 simulation ---
    # Demand uniformly distributed in [0,8]
    # Route size: 5
    # Overlap size: 1
    #results = simulate(scenario = 'baseline_k1', problem_sizes = [5,10,20,40,80], capacity = 20, route_size = 5, overlap_size = 1, cust_sims = 30, dem_sims = 200)

    # --- Short route simulation ---
    # Demand uniformly distributed in [0,8]
    # Route size: 2
    # Overlap size: 2
    #results = simulate(scenario = 'short_route', problem_sizes = [4,10,20,40,80], capacity = 8, route_size = 2, overlap_size = 2, cust_sims = 30, dem_sims = 200)

    # --- Long route simulation ---
    # Demand uniformly distributed in [0,8]
    # Route size: 10
    # Overlap size: 10
    #results = simulate(scenario = 'long_route', problem_sizes = [10,20,40,80], capacity = 40, route_size = 10, overlap_size = 10, cust_sims = 30, dem_sims = 200)

    # --- Stochastic customer simulation ---
    # Demand in {0,8} --> 0 w.p. 0.5 AND 8 w.p. 0.5
    # Route size: 5
    # Overlap size: 5
    #results = simulate(scenario = 'stochastic_customers', problem_sizes = [5,10,20,40,80], capacity = 20, route_size = 5, overlap_size = 5, cust_sims = 30, dem_sims = 200)

    # --- Binomial demand simulation ---
    # Demand ~ Binomial(8,0.5)
    # Route size: 5
    # Overlap size: 5
    #results = simulate(scenario = 'binomial', problem_sizes = [5,10,20,40,80], capacity = 20, route_size = 5, overlap_size = 5, cust_sims = 30, dem_sims = 200)

    # --- Low capacity simulation ---
    # Demand uniformly distributed in [0,8]
    # Route size: 5
    # Overlap size: 5
    # Capacity is 75% of E[D]
    #results = simulate(scenario = 'low_capacity', problem_sizes = [5,10,20,40,80], capacity = 15, route_size = 5, overlap_size = 5, cust_sims = 30, dem_sims = 200)

    # --- High capacity simulation ---
    # Demand uniformly distributed in [0,8]
    # Route size: 5
    # Overlap size: 5
    # Capacity is 125% of E[D]
    #results = simulate(scenario = 'high_capacity', problem_sizes = [5,10,20,40,80], capacity = 25, route_size = 5, overlap_size = 5, cust_sims = 30, dem_sims = 200)

    # Calculate summary statistics over instances
    means = results.groupby(['Scenario', 'Number of Customers', 'Routing Strategy', 'Metric'])['Value'].mean()
    sds = results.groupby(['Scenario', 'Number of Customers', 'Routing Strategy', 'Metric'])['Value'].std()
    ci_low = results.groupby(['Scenario', 'Number of Customers', 'Routing Strategy', 'Metric'])['Value'].quantile(0.025)
    ci_high = results.groupby(['Scenario', 'Number of Customers', 'Routing Strategy', 'Metric'])['Value'].quantile(0.975)

    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    outfile = 'output/results_{}.xlsx'.format(timestamp)

    with pd.ExcelWriter(outfile) as writer:
        results.to_excel(writer, sheet_name = 'baseline')
        #results.to_excel(writer, sheet_name='baseline_k3')
        #results.to_excel(writer, sheet_name='baseline_k1')
        #results.to_excel(writer, sheet_name='short_route')
        #results.to_excel(writer, sheet_name='long_route')
        #results.to_excel(writer, sheet_name='stochastic_customers')
        #results.to_excel(writer, sheet_name='binomial')
        #results.to_excel(writer, sheet_name='low_capacity')
        #results.to_excel(writer, sheet_name='high_capacity')
        means.to_excel(writer, sheet_name = 'summary_mean')
        sds.to_excel(writer, sheet_name = 'summary_sds')
        ci_low.to_excel(writer, sheet_name = 'summary_ci_low')
        ci_high.to_excel(writer, sheet_name = 'summary_ci_high')
