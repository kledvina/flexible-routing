import sys
import numpy as np
import pandas as pd
import math
import random
import time
from supporting import *

# GLOBAL VARIABLES
field_width = 100 # Customer location has x-coordinate in (0, field_width)
field_height = 100 # Customer location has y-coordinate in (0, field_height)
depot_x = 50 # Depot x-coordinate
depot_y = 50 # Depot y-coordinate


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
    pt, dt, ot, ft, rt, st = 0, 0, 0, 0, 0, 0

    # Create timestamp for backup outputs
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")

    # Print simulation parameters
    print('--- SIMULATION PARAMETERS ---')
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
            set_best_tours(row, primary_routes, extended_routes, capacity, route_size,
                           overlap_size)  # set cost-minimizing sequence
        pt += time.time() - new_pt

        # Loop through instances and find instance cost for different strategies

        for i in range(cust_sims):
            for j in range(dem_sims):

                # Get instance from array
                inst = instances[i][j]

                try:
                    # Solve dedicated routing
                    new_dt = time.time()
                    primary_routes = get_primary_routes(inst, route_size)
                    segments = create_full_trips(inst, primary_routes, capacity)
                    new_rows = create_report(inst, scenario, 'dedicated', segments)
                    sim_results = sim_results.append(new_rows, ignore_index=True)
                    dt += time.time() - new_dt

                    # Solve overlapped routing
                    new_ot = time.time()
                    primary_routes = get_primary_routes(inst, route_size)
                    extended_routes = get_extended_routes(inst, route_size, overlap_size)
                    segments = implement_k_overlapped_alg(inst, primary_routes, extended_routes, capacity, route_size, overlap_size)
                    new_rows = create_report(inst, scenario, 'overlapped', segments)
                    sim_results = sim_results.append(new_rows, ignore_index=True)
                    ot += time.time() - new_ot

                    # Solve fully flexible routing
                    new_ft = time.time()
                    segments = create_full_trips(inst, [inst.tour[1:]], capacity)
                    new_rows = create_report(inst, scenario, 'fully flexible', segments)
                    sim_results = sim_results.append(new_rows, ignore_index=True)
                    ft += time.time() - new_ft

                    # Solve reoptimization
                    new_rt = time.time()
                    cost, trips = solve_SDVRP(inst, capacity)
                    new_rows = pd.DataFrame([[scenario, inst.size, 'reoptimization', 'total cost', cost],
                                             [scenario, inst.size, 'reoptimization', 'trip count', trips]],
                                            columns=['Scenario', 'Number of Customers', 'Routing Strategy', 'Metric', 'Value'])
                    sim_results = sim_results.append(new_rows, ignore_index = True)
                    rt += time.time() - new_rt

                except Exception as e:
                    print('ERROR: {}'.format(e))
                    print('WARNING: Simulation failed to complete. Returning last Instance object.')
                    return inst

            # Save backup of data
            new_st = time.time()
            sim_results.to_csv('temp/backup_{}.csv'.format(timestamp))
            st += time.time() - new_st

            end = time.time()
            print('Customer instance {} complete. Time elapsed: {:.2f} s'.format(i + 1, end - start))

        print('Problems of size {} complete'.format(num_cust))

    print('Simulation complete.')
    print()
    print('--- RUNTIME BREAKDOWN ---')
    print('Setup: {:.2f} s'.format(pt))
    print('Dedicated: {:.2f} s'.format(dt))
    print('Overlapped: {:.2f} s'.format(ot))
    print('Full Flex.: {:.2f} s'.format(ft))
    print('Reoptimization: {:.2f} s'.format(rt))
    print('Saving: {:.2f} s'.format(st))

    return sim_results


#---------------------------------------------------------------------------------

if __name__ == "__main__":

    # Baseline simulation: demand uniformly distributed in [0,8]
    baseline_sim = simulate(scenario = 'baseline', problem_sizes = [80], capacity = 20, route_size = 5, overlap_size = 5, cust_sims = 10, dem_sims = 500)

    # Combine all simulation results into single dataframe
    combined = pd.concat([baseline_sim])

    # Calculate summary statistics over instances
    means = combined.groupby(['Scenario', 'Number of Customers', 'Routing Strategy', 'Metric'])['Value'].mean()
    sds = combined.groupby(['Scenario', 'Number of Customers', 'Routing Strategy', 'Metric'])['Value'].std()
    ci_low = combined.groupby(['Scenario', 'Number of Customers', 'Routing Strategy', 'Metric'])['Value'].quantile(0.025)
    ci_high = combined.groupby(['Scenario', 'Number of Customers', 'Routing Strategy', 'Metric'])['Value'].quantile(0.975)

    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    outfile = 'output/results_{}.xlsx'.format(timestamp)

    with pd.ExcelWriter(outfile) as writer:
        baseline_sim.to_excel(writer, sheet_name = 'baseline')
        means.to_excel(writer, sheet_name = 'summary_mean')
        sds.to_excel(writer, sheet_name = 'summary_sds')
        ci_low.to_excel(writer, sheet_name = 'summary_ci_low')
        ci_high.to_excel(writer, sheet_name = 'summary_ci_high')