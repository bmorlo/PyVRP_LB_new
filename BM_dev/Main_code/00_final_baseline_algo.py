import copy
import math
import random
import time
import re

import numpy as np
from scipy.spatial import distance as comp_dist

import matplotlib as mplb
from matplotlib import pyplot as plt

from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2

from pyvrp import (
    Model,
    SolveParams,
    PenaltyParams,
    PopulationParams,
)
from pyvrp.search import NeighbourhoodParams
from pyvrp.stop import MaxRuntime

# TODO:
"""
- Why not just use MULTI-DEPOT AND MULTI-PERIOD open (D)(C)VRP
- Why not just implement it as a multi-depot? Easily possible, but takes some more time...

- Generate e.g. one random test instance and save it. E.g. from the literature or from André. 
- Availabilities and locations added per week
- Change the availability matrix and observe the results on tour length
- Gaussche Verteilung: Fahrzeuge gehen einmal alle 8 Stunden kaputt --> die haben alle die gleiche Ausfallwahrscheinlichkeit
"""

# TODO:
"""
- Benchmarks for all the solvers (GUROBI (MTZ and also Dantzig-))
    --> Change the Distances to MANHATTEN!!!
    --> Use some kind of vehicle availability matrix over and over again!
    --> finalize the main_clean by putting all the solvers and running test instances 
    and recording the obj. value and the time (for 5 small and five large instances)
    --> Save everything in an Excel sheet
- Anzahl möglicher Lösungen bzw. Rechenzeit absehbar bei VRP bzw. bei meinem Code/ Solver -> wie ist das genau beim VRP?
- HGS in more detail! To get this to work better
"""


# TODO: Read the coords in from an instance when benchmarking!
def generate_coords_list(_num_coords_needed):
    """Returns a list of int lists specifying some random coordinates."""

    coords_list = []
    for i in range(_num_coords_needed):
        coords_single_random_node = [random.randint(-10000, 10000), random.randint(-10000, 10000)]
        coords_list.append(coords_single_random_node)

    return coords_list


# TODO: Cityblock-Distanzen für Benchmarks nutzen!
def generate_distance_matrix(_coords):
    """Returns a list of distance matrix rows aka the distance matrix belonging to the current problem instance."""

    distance_matrix = []
    for i in range(len(_coords)):
        frm_x = _coords[i][0]
        frm_y = _coords[i][1]
        temp_row_of_distance_matrix = []
        for j in range(len(_coords)):
            to_x = _coords[j][0]
            to_y = _coords[j][1]
            # Euklidean distances. Change it to 'Manhattan' for benchmarks.
            temp_row_of_distance_matrix.append(math.ceil(comp_dist.euclidean([frm_x, frm_y], [to_x, to_y])))
        distance_matrix.append(temp_row_of_distance_matrix)
        max(distance_matrix)

    return distance_matrix


def generate_vehicle_availability_matrix(_all_vehicle_ids, _campaign_length_in_weeks, _vehicle_breakdown_probability):
    """Creates the vehicle availability matrix for the given campaign length + 10."""
    vehicle_availability_matrix = {}
    for week_num in range(1, _campaign_length_in_weeks + 1 + 10):
        vehicle_availability_matrix[f'WEEK {week_num}'] = _all_vehicle_ids.copy()

        # Simulate vehicle breakdowns.
        for vehicle_id in _all_vehicle_ids:
            random_val_vehicle_breakdown = random.random()
            # Only try to remove a vehicle when there are more than 2 vehicles left.
            if (random_val_vehicle_breakdown < _vehicle_breakdown_probability
                    and len(vehicle_availability_matrix[f'WEEK {week_num}']) > 2):
                broken_down_vehicle_id = vehicle_id
                # Remove from the vehicle_availibility matrix.
                vehicle_availability_matrix[f'WEEK {week_num}'].remove(broken_down_vehicle_id)
    print(vehicle_availability_matrix)
    return vehicle_availability_matrix


# PyVRP parameters.
"""def pyvrp_parameters():
    PyVRP solver parameters for large scale VRP instances with short runtimes.

    Speficially:
    - The number of neighbours is decreased from 40 to 20 to minimize the
        number of potential local search moves per iteration. This improves
        convergence speed.
    - The maximum population size is decreased from 65 to 30 to minimize
        diversity and increase convergence speed.
    - The initial load penalty is increased from 20 to 1000 to encourage
        feasible solutions. This improves convergence speed and is most likely
        the reason why the default parameters do not work: a low load penalty
        encourages infeasible solutions. In combination with low runtimes, this
        can lead to the solver not finding a good feasible solution.

    neighbourhood = NeighbourhoodParams(nb_granular=20)
    population = PopulationParams(min_pop_size=10, generation_size=20)
    penalty = PenaltyParams(init_load_penalty=1000)

    return SolveParams(
        neighbourhood=neighbourhood,
        population=population,
        penalty=penalty,
    )
"""


# TODO: We probably do not need the dummy clients anymore, once we have implemented makespan.
def generate_optimized_routes_with_pyvrp(_max_runtime_in_seconds,
                                         _full_coords_list_dummy_clients_pyvrp,
                                         _full_distances_dummy_clients_pyvrp,
                                         _vehicle_capacities_dummy_clients_pyvrp,
                                         _full_demands_dummy_clients_pyvrp,
                                         _starting_points_indices,
                                         _end_points_indices, _functioning_vehicle_ids):
    """Generates optimized routes for the vehicles using the PyVRP solver. Outputs a list of int lists"""
    node_indices_all_tours_current_optimization_pyvrp = []
    pyvrp_model = Model()

    # From here on the magic happens... credits to Leon Lan from PyVRP for the idea behind this hack.
    # Create artificial depots for the vehicle types.
    # Each artificial depot is just a place-holder for the indices where a tour of a vehicle starts
    # and where it ends (indices of these locations are stored in the 'name' field).
    # The coords of the artificial depots do not matter!
    for k in range(len(_functioning_vehicle_ids)):
        start_k = _starting_points_indices[k]
        end_k = _end_points_indices[k]
        """Note that the x, y coords of the depots or clients are completely irrelevant! The solver solely relies 
        on the defined edges"""
        artificial_depot_k = pyvrp_model.add_depot(x=_full_coords_list_dummy_clients_pyvrp[0][0],
                                                   y=_full_coords_list_dummy_clients_pyvrp[0][1],
                                                   name=f"{start_k}_{end_k}")

        """Note that the index of the vehicle type will correspond to the correct vehicle position 
        in the functioning vehicle id list"""
        pyvrp_model.add_vehicle_type(num_available=1, capacity=_vehicle_capacities_dummy_clients_pyvrp[k],
                                     depot=artificial_depot_k,
                                     name=_functioning_vehicle_ids[k])

    # This adds all real locations and additionally the dummy clients to the model
    # A location idx of 0 is equal to real_coords index of len(_starting_points_indices) + 1.
    # These location indices are what is written in the client names each time.
    for location_idx in range(len(_starting_points_indices) + 1, len(_full_coords_list_dummy_clients_pyvrp)):
        pyvrp_model.add_client(x=_full_coords_list_dummy_clients_pyvrp[location_idx][0],
                               y=_full_coords_list_dummy_clients_pyvrp[location_idx][1],
                               delivery=_full_demands_dummy_clients_pyvrp[location_idx],
                               name=f"{location_idx}_{location_idx}")

    # The locations are the artificial depots and the clients, NOT the end depot.
    # We abuse an artificial depot to assign the 'leaving' distances (to clients) to it that would normally occur
    # when starting at the specified starting point. We also assign the "reaching" distances (to clients) to it
    # that would normally occur when ending a tour at the end depot.

    """Note that the indices in the from names can actually be regarded as real world coords indices"""
    first_dummy_client_location_index = len(_full_coords_list_dummy_clients_pyvrp) - len(_starting_points_indices)
    # If you subtract those two lengths you get a number equal to end-depot + artificial depots + normal clients.
    # Since the location indices that were stored in the names correspond to the real coord indices,
    # the first dummy client can be accessed by this very length value (since index = length - 1).

    for frm in pyvrp_model.locations:
        for to in pyvrp_model.locations:
            # Note that the name has the "external" indices, NOT the internal PyVRP index.
            # For (dummy) clients, the index is the same in both segments of the name.
            # For artificial depots, the first segment is the index of the real coords location that is actually left
            # when the artificial depot is referenced as a frm-node.
            # The second segment corresponds to the real coords location that is actually visited when the artificial
            # depot is referenced as a to-node.

            location_that_is_left_idx = int(frm.name.split("_")[0])
            location_that_is_visited_idx = int(to.name.split("_")[1])

            if location_that_is_visited_idx >= first_dummy_client_location_index:
                # A dummy client was reached. If it is left, it acts completely normal --> see "else".
                if location_that_is_left_idx < first_dummy_client_location_index:
                    # The dummy client was reached by the correct/corresponding artificial depot.
                    if (location_that_is_left_idx + (
                            len(_full_coords_list_dummy_clients_pyvrp) - len(_starting_points_indices) - 1)
                            == location_that_is_visited_idx):
                        # Make it a no-brainer to immediately visit it. We need the additional - 1!
                        pyvrp_model.add_edge(frm, to, distance=0)
                        # We simply disregard any edges from other artificial depots to this dummy client!
            else:
                # A normal client, or a depot was reached. Depots should not be able to reach any client directly.
                # --> depot to xy = 0
                # Thus, only set their closed loop edges.
                # Depots should only be reached via "zero distance".
                # Thus, for all the locations you are coming from, depot, client, or dummy client,
                # set the edge back to the depot to zero.
                if location_that_is_visited_idx == 0:
                    # A depot was reached.
                    pyvrp_model.add_edge(frm, to, distance=0)
                elif location_that_is_left_idx >= len(_starting_points_indices) - 1:
                    # A client or dummy client was reached. ONLY SET THE EDGES THAT DO NOT START FROM THE DEPOTS.
                    # Depots only visit dummy-clients.
                    correct_distance = _full_distances_dummy_clients_pyvrp[location_that_is_left_idx][
                        location_that_is_visited_idx] if frm != to else 0
                    pyvrp_model.add_edge(frm, to, distance=correct_distance)

    result_pyvrp = pyvrp_model.solve(stop=MaxRuntime(_max_runtime_in_seconds), seed=42,
                                     display=True)
    assert result_pyvrp.is_feasible()

    for route in result_pyvrp.best.routes():
        # Let's map it back to the real depot and client indices.
        start = _starting_points_indices[route.vehicle_type()]
        end = _end_points_indices[route.vehicle_type()]

        # PyVRP, internally constructs its own indices of the locations (== artificial depots + clients). As a result,
        # the indices of the clients in a tour are lower by exactly one (the end depot comes first in my own indexing).
        # Additionally, we get rid of the dummy_clients again!
        # internal_clients = [idx + 1 for idx in route.visits()]
        # print(f"Clients\n{internal_clients}")
        clients = [idx + 1 for idx in route.visits() if (idx + 1) < first_dummy_client_location_index]
        # print(f"Actual clients\n{clients}")
        node_indices_all_tours_current_optimization_pyvrp.append([start] + clients + [start])

    print(f"Objective value PyVRP: {result_pyvrp.cost()}")

    # The latter returns the vehicle ids for double-checking purposes.
    return node_indices_all_tours_current_optimization_pyvrp, [str(vehicle_type) for vehicle_type in
                                                               pyvrp_model.vehicle_types]


def generate_optimized_routes_with_or_routing(_max_runtime_in_seconds,
                                              _full_coords_list,
                                              _full_distances,
                                              _vehicle_capacities,
                                              _full_demands,
                                              _starting_points_indices,
                                              _end_points_indices, _functioning_vehicle_ids):
    """Generates optimized routes for the vehicles using the OR Routing solver. Outputs a list of int lists"""
    node_indices_all_tours_current_optimization_or_routing = []

    # Create the routing index manager.
    routing_manager = pywrapcp.RoutingIndexManager(len(_full_distances), len(_functioning_vehicle_ids),
                                                   _starting_points_indices, _end_points_indices)
    # Create Routing Model.
    routing_model = pywrapcp.RoutingModel(routing_manager)

    # Create and register a transit callback.
    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = routing_manager.IndexToNode(from_index)
        to_node = routing_manager.IndexToNode(to_index)
        return _full_distances[from_node][to_node]

    transit_callback_index = routing_model.RegisterTransitCallback(distance_callback)
    # Define cost of each arc.
    routing_model.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    def demand_callback(from_index):
        """Returns the demand of the node."""
        # Convert from routing variable Index to demands NodeIndex.
        from_node = routing_manager.IndexToNode(from_index)
        return _full_demands[from_node]

    demand_callback_index = routing_model.RegisterUnaryTransitCallback(demand_callback)
    routing_model.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,  # null capacity slack
        _vehicle_capacities,  # vehicle maximum capacities
        True,  # start cumul to zero
        "Capacity",
    )

    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    # First solution strategy.
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    # Local search strategy.
    search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    # Other parameters.
    search_parameters.time_limit.FromSeconds(_max_runtime_in_seconds)
    # search_parameters.solution_limit = 50000
    search_parameters.log_search = False

    # Solve the problem.
    solution = routing_model.SolveWithParameters(search_parameters)

    # Makes sure that the problem is not infeasible!
    assert routing_model.status() != 6

    # Print solution on console.
    print(f"Objective value OR_Routing: {solution.ObjectiveValue()}")

    total_distance = 0
    for vehicle_idx in range(len(_functioning_vehicle_ids)):
        index = routing_model.Start(vehicle_idx)
        # plan_output = f"Route for vehicle {vehicle_id}:\n"
        route_distance = 0
        tour_k_indices = []
        while not routing_model.IsEnd(index):
            node_index = routing_manager.IndexToNode(index)
            # plan_output += f" {node_index} -> "
            tour_k_indices.append(node_index)
            previous_index = index
            index = solution.Value(routing_model.NextVar(index))
            route_distance += routing_model.GetArcCostForVehicle(
                previous_index, index, vehicle_idx
            )
        # This prints the last node I think!
        end_index = routing_manager.IndexToNode(index)
        # plan_output += f" {end_index}\n"
        tour_k_indices.append(end_index)
        # plan_output += f"Distance of the route: {route_distance}\n"
        # print(plan_output)
        total_distance += route_distance
        node_indices_all_tours_current_optimization_or_routing.append(tour_k_indices)

    return node_indices_all_tours_current_optimization_or_routing


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    """Helps with sorting string lists according to their numbers within the strings."""
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [atoi(c) for c in re.split(r'(\d+)', text)]


# TODO: Adjust figure axes with respect to the min, max coordinate values of an instance.
def plot_results_euklid(_node_indices_all_tours_current_optimization, _functioning_vehicle_ids_this_week,
                        _vehicle_capacities, _full_coords_list, _vehicle_id_to_color_dictionary, _all_vehicle_ids,
                        _temp_coords_final_routes, _order_horizon, _campaign_length_in_weeks,
                        _vehicle_capacity, _optimization_idx):
    """Plots the optimized routes in an interactive fashion with index annotations. The gray dot represents the
    vehicle"""

    # Run GUI event loop.
    plt.ion()
    figure, ax = plt.subplots(figsize=(8, 6))

    # Sort the vehicle id list. Currently unused. For this to work you would have to make the starting locations a dict.
    # temp_vehicle_ids_this_week = _functioning_vehicle_ids_this_week.copy()
    # temp_vehicle_ids_this_week.sort(key=natural_keys)

    figure.suptitle(
        f'Hindsight view of week {_optimization_idx} of a {_campaign_length_in_weeks}-week DC campaign.\n'
        f'This week, vehicles {_functioning_vehicle_ids_this_week} were operational.\n'
        f'Employed order horizon: {_order_horizon} weeks.\n'
        f'Applied capacity of a vehicle: {_vehicle_capacity} POI(s).', fontsize=12)

    figure.tight_layout()
    ax.set_xlim(xmin=-11000, xmax=11000)
    ax.set_ylim(ymin=-11000, ymax=11000)
    ax.set_axisbelow(True)

    # Plot the end depot beforehand.
    end_depot_coords = _full_coords_list[0]
    plt.plot(end_depot_coords[0], end_depot_coords[1], 'ks', markersize=16,
             zorder=(_vehicle_capacities[0] + 3) * len(_functioning_vehicle_ids_this_week),
             label='Main depot (homebase)')

    # Note that the functioning vehicle ids are in the same order, as the route indices k. This is,
    # because in each optimization, we give the pyvrp solver the ids in the order that they are specified
    # in the functioning vehicle list.
    for k in range(len(_functioning_vehicle_ids_this_week)):
        legend_counter_vehicle = 0
        indices_tour_k = np.array(_node_indices_all_tours_current_optimization[k])
        coords_as_array = np.array(_full_coords_list)
        coords_tour_k = coords_as_array[indices_tour_k]
        x, y = zip(*coords_tour_k)
        for i in range(len(x)):
            if i == 0 and _optimization_idx != 1:
                plt.plot(x[i], y[i], 's', color=_vehicle_id_to_color_dictionary[_functioning_vehicle_ids_this_week[k]],
                         markersize=16,
                         label=f'Temporary depot {_functioning_vehicle_ids_this_week[k]}')
            if i != len(x) - 1:
                # Do not plot the way back to the depot! We are working with an Open MD-DCVRP.
                if legend_counter_vehicle == 0:
                    plt.plot(x[0:i + 1], y[0:i + 1], 'o--',
                             color=_vehicle_id_to_color_dictionary[_functioning_vehicle_ids_this_week[k]],
                             markersize=10, label=f'Planned tour {_functioning_vehicle_ids_this_week[k]}')
                    legend_counter_vehicle += 1
                else:
                    plt.plot(x[0:i + 1], y[0:i + 1], 'o--',
                             color=_vehicle_id_to_color_dictionary[_functioning_vehicle_ids_this_week[k]],
                             markersize=10)

    # Plot the final, actually driven tour_segments in black.""
    legend_counter = 0
    for vehicle_id in _all_vehicle_ids:
        coords_final_tour_vehicle = np.array(_temp_coords_final_routes[vehicle_id])
        x_final, y_final = zip(*coords_final_tour_vehicle)
        for i in range(len(x_final)):
            if legend_counter == 0:
                # Only do this once for all the plots.
                plt.plot(x_final[0:i + 1], y_final[0:i + 1], 'ks-', markersize=6, linewidth=2,
                         label='Driven tours')
                legend_counter += 1
            else:
                plt.plot(x_final[0:i + 1], y_final[0:i + 1], 'ks-', markersize=6, linewidth=2)

            if i == len(x_final) - 1:
                # Only annotate the current position.
                ax.annotate(f"{vehicle_id}", (x_final[i] + 250, y_final[i] + 250), fontsize=8)
            # Drawing updated values
            figure.canvas.draw()
            figure.canvas.flush_events()
            time.sleep(0.2)

    plt.legend(framealpha=0.1)
    plt.show(block=True)


def main():
    """Run the optimizations for a specified order horizon length. Currently we would work with
    an order horizon of at least one week (then this would be NN). This way, we always start with a
    "full house", then visit some locations (which we remove if successfully visited) and only then,
    AFTER that, we add new locations that we could visit"""

    # Set the time that the solver has to get to a solution.
    max_runtime_in_seconds = 30

    # Define the problem instance
    num_theoretically_available_vehicles = 5
    break_down_probability = 0.2
    campaign_length_in_weeks = 8
    order_horizon = 3
    pre_campaign_pool_size = order_horizon * num_theoretically_available_vehicles

    all_vehicle_ids = [f'vehicle-{i}' for i in range(1, num_theoretically_available_vehicles + 1)]
    # Define the vehicle availabilities over the course of the campaign and beyond (+ 10 weeks).
    vehicle_availability_matrix = generate_vehicle_availability_matrix(all_vehicle_ids, campaign_length_in_weeks,
                                                                       break_down_probability)

    # Depending on its vehicle ID, set the color of a tour in the plots
    color_vector = np.linspace(0.1, 0.9, len(all_vehicle_ids))
    cmap = mplb.colormaps['turbo']
    all_vehicle_colors = []
    for color_float in color_vector:
        color_rgba = cmap(color_float)
        color_hex = mplb.colors.to_hex(color_rgba, keep_alpha=True)
        all_vehicle_colors.append(color_hex)
    vehicle_id_to_color_dictionary = dict(zip(all_vehicle_ids, all_vehicle_colors))

    # Initializations.
    functioning_vehicle_ids_this_week = vehicle_availability_matrix['WEEK 1']

    coords_end_depot = [0, 0]
    starting_point_indices = list(range(1, len(functioning_vehicle_ids_this_week) + 1))
    end_points_indices = [0] * len(functioning_vehicle_ids_this_week)
    starting_points_coords = [coords_end_depot.copy()] * len(functioning_vehicle_ids_this_week)

    poi_pool = generate_coords_list(pre_campaign_pool_size)
    full_coords_list = [coords_end_depot.copy()] + starting_points_coords + poi_pool
    full_coords_list_dummy_clients_pyvrp = full_coords_list + starting_points_coords
    full_distances = generate_distance_matrix(full_coords_list)
    full_distances_dummy_clients_pyvrp = generate_distance_matrix(full_coords_list_dummy_clients_pyvrp)

    # Create the vehicle capacities.
    vehicle_capacities = ([math.ceil(len(poi_pool) / len(functioning_vehicle_ids_this_week))]
                          * len(functioning_vehicle_ids_this_week))
    vehicle_capacities_dummy_clients_pyvrp = ([math.ceil((len(poi_pool) + len(starting_point_indices))
                                                         / len(functioning_vehicle_ids_this_week))]
                                              * len(functioning_vehicle_ids_this_week))

    # Create demands that correspond to 1) the end depot, 2) the starting points and 3) the points of interest.
    full_demands = [0] + [0] * len(functioning_vehicle_ids_this_week) + [1] * len(poi_pool)
    full_demands_dummy_clients_pyvrp = ([0] + [0] * len(functioning_vehicle_ids_this_week)
                                        + [1] * len(poi_pool) + [1] * len(
                functioning_vehicle_ids_this_week))

    # Dictionary saves the final routes accessible via the ids as keys. This is needed, because from optimization to
    # optimization the order of the functioning vehicle IDs and thus the order of the routes of an optimization
    # step are subject to change.
    coords_final_routes = {}
    for _id_ in all_vehicle_ids:
        coords_final_routes[_id_] = [coords_end_depot.copy()]

    currently_driven_total_transfer_drive_kilometers_of_the_campaign = 0
    final_optimization_step = False
    optimization_idx = 1

    """Note: In each iteration/week the order of the functioning vehicle ids corresponds to the order of the 
            used vehicles in PyVRP."""
    # Rerun the optimization in a rolling-horizon kind of fashion.
    while poi_pool:
        routes_indices_pyvrp, details_of_pyvrp_vehicle_types = generate_optimized_routes_with_pyvrp(
            max_runtime_in_seconds, full_coords_list_dummy_clients_pyvrp, full_distances_dummy_clients_pyvrp,
            vehicle_capacities_dummy_clients_pyvrp, full_demands_dummy_clients_pyvrp, starting_point_indices,
            end_points_indices, functioning_vehicle_ids_this_week)

        # Note:
        # OR-Tools neglects the last tour segment back to the depot in an EMPTY tour.
        # As a result, its planned tours can differ from pyvrp in certain scenarios.
        """
        routes_indices_or_routing = generate_optimized_routes_with_or_routing(
            max_runtime_in_seconds, full_coords_list, full_distances, vehicle_capacities,
            full_demands, starting_point_indices, end_points_indices, functioning_vehicle_ids_this_week)
        """

        print(f"\nTours PyVRP in iteration {optimization_idx}. Please note that the starting point indices "
              f"show a value != 0 even if the starting point was the end-depot. They also do not indicate the vehicle id")
        for route_index in range(len(functioning_vehicle_ids_this_week)):
            print(
                f"Tour of {functioning_vehicle_ids_this_week[route_index]}: {routes_indices_pyvrp[route_index]}")
            print([full_coords_list_dummy_clients_pyvrp[coord_i] for coord_i in routes_indices_pyvrp[route_index]])

        """
        print(f"\nTours OR Routing in iteration {optimization_idx}. Please note that the starting point indices "
              f"show a value != 0 even if the starting point was the end-depot. They also do not indicate the vehicle id")
        for route_index in range(len(functioning_vehicle_ids_this_week)):
            print(f"Tour of {functioning_vehicle_ids_this_week[route_index]}: {routes_indices_or_routing[route_index]}")
            print([full_coords_list[coord_i_or] for coord_i_or in routes_indices_or_routing[route_index]])
        """

        # Set the functioning_vehicle_ids for NEXT WEEK:
        functioning_vehicle_ids_next_week_theoretically = vehicle_availability_matrix[f'WEEK {optimization_idx + 1}']

        # The temp_coords_final_routes makes sure we do not show the route back to the main depot
        # straight-away in our plots whenever a vehicle breaks down.
        temp_coords_final_routes = copy.deepcopy(coords_final_routes)

        # Update the starting points and record what has just happened.
        starting_points_coords = []
        broken_down_vehicles_this_week = []

        """The most important for-loop. It updates the starting points and records the travelled distance!"""
        # Index the previous functioning vehicle ids list.
        for route_index in range(len(functioning_vehicle_ids_this_week)):
            # Access the old starting position of a vehicle via index 0 in a route.
            left_location_index = routes_indices_pyvrp[route_index][0]
            # Access the new, current location of a vehicle via an index 1 in a route.
            visited_location_index = routes_indices_pyvrp[route_index][1]
            # Add the transfer drive kilometers. Note that despite using dummy clients, we use the actual distances.
            currently_driven_total_transfer_drive_kilometers_of_the_campaign += full_distances[left_location_index][
                visited_location_index]
            # Record the visited coords for the final tour.
            coords_final_routes[functioning_vehicle_ids_this_week[route_index]].append(
                full_coords_list[visited_location_index])
            temp_coords_final_routes[functioning_vehicle_ids_this_week[route_index]].append(
                full_coords_list[visited_location_index])

            # Only add a visited location as a starting point if the vehicle is still part of the
            # functioning vehicle ids list OF THE NEXT WEEK!
            if functioning_vehicle_ids_this_week[route_index] in functioning_vehicle_ids_next_week_theoretically:
                # Set the new starting point.
                starting_points_coords.append(full_coords_list[visited_location_index])
                # Remove location from actual poi pool!
                location_in_question = full_coords_list[visited_location_index]
                # Do not try to remove the home-base because it is not part of the poi pool!
                if visited_location_index != 0:
                    poi_pool.remove(location_in_question)

            else:  # Vehicle broke down and is immediately sent back to the depot.
                # Add the distance of the return to the end-depot to the total transfer drive KMs.
                currently_driven_total_transfer_drive_kilometers_of_the_campaign += \
                    full_distances[visited_location_index][0]
                # Record the depot coords.
                coords_final_routes[functioning_vehicle_ids_this_week[route_index]].append(coords_end_depot.copy())
                # Record the broken-down vehicle ids.
                broken_down_vehicles_this_week.append(functioning_vehicle_ids_this_week[route_index])

        # Only plot at this point using the temp_coords_final_routes list to avoid add
        plot_results_euklid(routes_indices_pyvrp, functioning_vehicle_ids_this_week,
                            vehicle_capacities, full_coords_list,
                            vehicle_id_to_color_dictionary, all_vehicle_ids, temp_coords_final_routes,
                            order_horizon, campaign_length_in_weeks, vehicle_capacities[0],
                            optimization_idx)

        # Check for new incoming functioning vehicle ids. And append the depot to the starting points list.
        new_vehicle_ids_incoming = [id_new for id_new in functioning_vehicle_ids_next_week_theoretically if
                                    id_new not in functioning_vehicle_ids_this_week]

        # Use a new functioning_vehicle_ids list to set up the next iteration optimization iteration.
        # Put the incoming vehicle ids at the end.
        for id_to_remove in broken_down_vehicles_this_week:
            functioning_vehicle_ids_this_week.remove(id_to_remove)

        # Update the starting points and the functioning vehicle ids.
        functioning_vehicle_ids = functioning_vehicle_ids_this_week + new_vehicle_ids_incoming
        for idx in range(len(new_vehicle_ids_incoming)):
            starting_points_coords.append(coords_end_depot.copy())

        # TODO: read new locations from an instance...
        # End of campaign:
        # The last time we add new locations to the poi pool after an optimization is
        # in the week/iteration No. (campaign_length_in_weeks - order_horizon)
        if optimization_idx <= campaign_length_in_weeks - order_horizon:
            # Add the plannned locations that are the t.o.b.h. ahead
            new_locations = generate_coords_list(num_theoretically_available_vehicles)
            poi_pool += new_locations

        # If the poi pool remains empty after disruptions & dynamics it means you have just visited
        # your last location and nothing bad has happened. But the vehicles still need to drive home!
        if not poi_pool and not final_optimization_step:
            poi_pool = [coords_end_depot.copy()] * len(functioning_vehicle_ids)
            final_optimization_step = True

        # Update the starting point indices and the number of endpoints.
        starting_point_indices = list(range(1, len(functioning_vehicle_ids) + 1))
        end_points_indices = [0] * len(functioning_vehicle_ids)

        # Update the coords list and recalculate the distance matrix.
        full_coords_list = [coords_end_depot.copy()] + starting_points_coords + poi_pool
        full_coords_list_dummy_clients_pyvrp = ([coords_end_depot.copy()] + starting_points_coords
                                                + poi_pool + starting_points_coords)

        full_distances = generate_distance_matrix(full_coords_list)
        full_distances_dummy_clients_pyvrp = generate_distance_matrix(full_coords_list_dummy_clients_pyvrp)

        # Update the vehicle capacities and the demands.
        vehicle_capacities = ([math.ceil(len(poi_pool) / len(functioning_vehicle_ids))]
                              * len(functioning_vehicle_ids))
        vehicle_capacities_dummy_clients_pyvrp = ([math.ceil(
            (len(poi_pool) + len(starting_point_indices)) / len(functioning_vehicle_ids))]
                                                  * len(functioning_vehicle_ids))
        full_demands = [0] + [0] * len(functioning_vehicle_ids) + [1] * len(poi_pool)
        full_demands_dummy_clients_pyvrp = ([0] + [0] * len(functioning_vehicle_ids)
                                            + [1] * len(poi_pool) + [1] * len(functioning_vehicle_ids))

        functioning_vehicle_ids_this_week = functioning_vehicle_ids.copy()

        optimization_idx += 1
        print(
            f"Current transfer drive kilometers: {currently_driven_total_transfer_drive_kilometers_of_the_campaign}\n")


if __name__ == '__main__':
    main()
