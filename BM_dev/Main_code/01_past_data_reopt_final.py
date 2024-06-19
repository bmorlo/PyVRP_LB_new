"""This is a revised version of the baseline reopt algorithm.
Now we are using actual distances and read in the past data"""
import copy
import math
import json
import numpy as np
import pandas as pds

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

import folium
from folium.plugins import Fullscreen
import matplotlib as mplb



def read_inputs():
    """Reads the inputs from the central Excel file."""
    df_vehicle_ids = pds.read_excel('BM_dev/Main_code/Central_Input_Data_Andre.xlsx', sheet_name='Vehicle_ids')
    df_locations_and_coords = pds.read_excel('BM_dev/Main_code/Central_Input_Data_Andre.xlsx', sheet_name='Location_names_to_coords')
    df_planned_locations = pds.read_excel('BM_dev/Main_code/Central_Input_Data_Andre.xlsx', sheet_name='Planned_locations')
    df_vehicle_availability_matrix = pds.read_excel('BM_dev/Main_code/Central_Input_Data_Andre.xlsx',
                                                    sheet_name='Vehicle_availability_matrix')
    all_location_names = df_locations_and_coords['NAME'].tolist()

    # Create a dictionary that maps a name to the coordinates.
    temp_row_index_location_name_to_coords = 0
    location_name_to_coords_dict = {}
    for location in df_locations_and_coords['NAME']:
        location_name_to_coords_dict[location] = [
            df_locations_and_coords['LAT'][temp_row_index_location_name_to_coords].item(),
            df_locations_and_coords['LONG'][temp_row_index_location_name_to_coords].item()]
        temp_row_index_location_name_to_coords += 1

    dist_dict = {}
    geo_dict = {}
    # Add the dist and geo rows as sub-dicts to their respective outer dicts.
    for ITER_IDX in range(len(all_location_names)):
        with open(f'BM_dev/Main_code/json_files_for_dist_and_geo_matrices/revised_distance_matrix_row_'
                  f'{all_location_names[ITER_IDX]}.json') as dismarow:
            dist_row = json.load(dismarow)
        dist_dict.update(dist_row)
        with open(f'BM_dev/Main_code/json_files_for_dist_and_geo_matrices/revised_geometry_matrix_row_'
                  f'{all_location_names[ITER_IDX]}.json') as geomarow:
            geo_row = json.load(geomarow)
        geo_dict.update(geo_row)

    # Make the dist_matrix and the geo_matrix symmetric!
    for frm_idx in range(len(all_location_names)):
        for to_idx in range(len(all_location_names)):
            if frm_idx < to_idx:
                # Distances.
                dist_dict[all_location_names[frm_idx]][all_location_names[to_idx]] = (
                    dist_dict)[all_location_names[to_idx]][all_location_names[frm_idx]]
                # Geodatas.
                temp_matrix_value_geo_dict = (
                    geo_dict)[all_location_names[to_idx]][all_location_names[frm_idx]]
                temp_matrix_value_geo_dict['coordinates'] = temp_matrix_value_geo_dict['coordinates'][::-1]
                geo_dict[all_location_names[frm_idx]][all_location_names[to_idx]] = temp_matrix_value_geo_dict

    return (df_vehicle_ids, df_locations_and_coords, df_planned_locations, df_vehicle_availability_matrix, dist_dict,
            geo_dict, location_name_to_coords_dict)


def generate_distance_matrix(_locations, _dist_dict):
    """Returns a list of distance matrix rows aka the distance matrix belonging to the current problem instance."""

    distance_matrix = []
    for i in range(len(_locations)):
        frm_name = _locations[i]
        row_i_of_distance_matrix = []
        for j in range(len(_locations)):
            to_name = _locations[j]
            dist_tour_segment = _dist_dict[frm_name][to_name]
            row_i_of_distance_matrix.append(dist_tour_segment)
        distance_matrix.append(row_i_of_distance_matrix)
    return distance_matrix

def generate_optimized_routes_with_pyvrp(_max_runtime_in_seconds,
                                         _full_coords_list_dummy_clients_pyvrp,
                                         _full_distances_dummy_clients_pyvrp,
                                         _vehicle_capacities_dummy_clients_pyvrp,
                                         _full_demands_dummy_clients_pyvrp,
                                         _starting_points_indices,
                                         _end_points_indices, _functioning_vehicle_ids, _seed):
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
        """Note that the x, y coords of the depots or clients are completely irrelevant!"""
        artificial_depot_k = pyvrp_model.add_depot(x=_full_coords_list_dummy_clients_pyvrp[0][0],
                                                   y=_full_coords_list_dummy_clients_pyvrp[0][1],
                                                   name=f"{start_k}_{end_k}")

        """Note that the index of the vehicle type will correspond to the correct position of the vehicle id in
        in the functioning vehicle id list."""
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
    # Since the location indices that were stored in the names correspond to the real coord indices, the first dummy
    # client can be accessed by this very length value (since index = length - 1).

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

                # TODO: Maybe specify those edges but not with such a high distance value!
            else:
                # A normal client, or a depot was reached. Depots should not be able to reach any client directly.
                # Depots should only be able to reach dummy clients directly!
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

    result_pyvrp = pyvrp_model.solve(stop=MaxRuntime(_max_runtime_in_seconds), seed=_seed,
                                     display=False)
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
        node_indices_all_tours_current_optimization_pyvrp.append([start] + clients + [end])

    # print(f"Objective value PyVRP: {result_pyvrp.cost()}")
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


def main(_proposed_seed, _proposed_order_horizon):
    """Run the optimizations for a specific order horizon length"""

    # Read in the central inputs.
    (df_vehicle_ids, df_locations_and_coords, df_planned_locations,
     df_vehicle_availability_matrix, dist_dict, geo_dict, name_to_coords_dict) = read_inputs()

    # Define the problem instance.
    num_theoretically_available_vehicles = len(df_vehicle_ids['ID'].tolist())
    campaign_length_in_weeks = 20
    order_horizon = _proposed_order_horizon
    pre_campaign_pool_size = order_horizon * num_theoretically_available_vehicles
    all_vehicle_ids = df_vehicle_ids['ID'].tolist()
    # Vehicle Availability Matrix needs dropna() since some columns of the pandas dataframe are not completely filled.
    functioning_vehicle_ids_this_week = (df_vehicle_availability_matrix['WEEK 1'].dropna()).tolist()

    color_vector = np.linspace(0.1, 0.9, len(all_vehicle_ids))
    cmap = mplb.colormaps['turbo']
    all_vehicle_colors = []
    for color_float in color_vector:
        color_rgba = cmap(color_float)
        color_hex = mplb.colors.to_hex(color_rgba, keep_alpha=True)
        all_vehicle_colors.append(color_hex)

    # Initialize some locations lists.
    homebase = df_locations_and_coords['NAME'][0]
    starting_point_indices = list(range(1, len(functioning_vehicle_ids_this_week) + 1))
    end_points_indices = [0] * len(functioning_vehicle_ids_this_week)
    starting_points = [homebase] * len(functioning_vehicle_ids_this_week)

    # Generate the pre-campaign poi pool.
    poi_pool = df_planned_locations['WEEK 1'].dropna().tolist()
    for week_num in range(2, order_horizon + 1):
        poi_pool += df_planned_locations[f'WEEK {week_num}'].dropna().tolist()

    # Initialize some more locations lists.
    full_locations_list = [homebase] + starting_points + poi_pool
    full_locations_list_dummy_clients_pyvrp = [homebase] + starting_points + poi_pool + starting_points
    full_distances = generate_distance_matrix(full_locations_list, dist_dict)
    full_distances_dummy_clients_pyvrp = generate_distance_matrix(full_locations_list_dummy_clients_pyvrp, dist_dict)

    # Initialize the vehicle capacities.
    vehicle_capacities = ([math.ceil(len(poi_pool) / len(functioning_vehicle_ids_this_week))]
                          * len(functioning_vehicle_ids_this_week))
    vehicle_capacities_dummy_clients_pyvrp = (
            [math.ceil(
                (len(poi_pool) + len(functioning_vehicle_ids_this_week)) / len(
                    functioning_vehicle_ids_this_week))]
            * len(functioning_vehicle_ids_this_week))

    # Create demands that correspond to 1) the end depot, 2) the starting points,
    # 3) the points of interest [and 4) the dummy clients].
    full_demands = [0] + [0] * len(functioning_vehicle_ids_this_week) + [1] * len(poi_pool)
    full_demands_dummy_clients_pyvrp = (
            [0] + [0] * len(functioning_vehicle_ids_this_week) + [1] * len(poi_pool)
            + [1] * len(functioning_vehicle_ids_this_week))
    
    # Set the time that the solver has to get to a solution.
    max_runtime_in_seconds = (len(poi_pool) + len(functioning_vehicle_ids_this_week))*2.08

    # Dictionary saves the final routes accessible via the ids as keys. This is needed, because from optimization to
    # optimization the order of the functioning vehicle IDs and thus the order of the routes of an optimization
    # step are subject to change.
    locations_in_final_routes = {}
    for _id_ in all_vehicle_ids:
        locations_in_final_routes[_id_] = [homebase]

    currently_driven_total_transfer_drive_kilometers_of_the_campaign = 0
    final_optimization_step = False
    optimization_idx = 1

    """Note: In each iteration the order of the functioning vehicle ids corresponds to the order of the 
            used vehicles in PyVRP."""
    # Rerun the optimization in a rolling-horizon kind of fashion. But stop the vehicles in the field
    # after the campaign length.
    while poi_pool and optimization_idx <= campaign_length_in_weeks:
        full_coords_list_dummy_clients_pyvrp = [name_to_coords_dict[name] for name in
                                                full_locations_list_dummy_clients_pyvrp]
        full_coords_list_dummy_clients_pyvrp = [[math.ceil(lat), math.ceil(long)] for lat, long in
                                                full_coords_list_dummy_clients_pyvrp]
        routes_indices_pyvrp, details_of_pyvrp_vehicle_types = generate_optimized_routes_with_pyvrp(
            max_runtime_in_seconds, full_coords_list_dummy_clients_pyvrp, full_distances_dummy_clients_pyvrp,
            vehicle_capacities_dummy_clients_pyvrp, full_demands_dummy_clients_pyvrp, starting_point_indices,
            end_points_indices, functioning_vehicle_ids_this_week, _seed=_proposed_seed)

        # OR-Tools neglects the last tour segment back to the depot in an EMPTY tour.
        # As a result, its planned tours can differ from pyvrp in certain scenarios.
        # full_coords_list = [name_to_coords_dict[name] for name in full_locations_list]
        # full_coords_list = [[math.ceil(lat), math.ceil(long)] for lat, long in full_coords_list]
        # routes_indices_or_routing = generate_optimized_routes_with_or_routing(
        # max_runtime_in_seconds, full_coords_list, full_distances, vehicle_capacities,
        # full_demands, starting_point_indices, end_points_indices, functioning_vehicle_ids_this_week)

        """
        print(f"\nTours PyVRP in iteration {optimization_idx}.")
        for route_index in range(len(functioning_vehicle_ids_this_week)):
            temp_planned_tour = [full_locations_list_dummy_clients_pyvrp[loc_idx] for loc_idx in
                                 routes_indices_pyvrp[route_index]]
            print(f"Planned tour of vehicle {functioning_vehicle_ids_this_week[route_index]}: {temp_planned_tour}\n")
        """

        # print(f"\nTours OR Routing in iteration {optimization_idx}. Please note that the starting point indices "
        #       f"show a value != 0 even if the starting point was the end-depot. They also do not indicate the vehicle id")
        # for route_index in range(len(functioning_vehicle_ids_this_week)):
        # print(routes_indices_or_routing[route_index])
        #     print([full_locations_list[loc_idx_or] for loc_idx_or in routes_indices_or_routing[route_index]])

        # Set the functioning_vehicle_ids_this_week for NEXT WEEK:
        functioning_vehicle_ids_next_week_theoretically = (
            df_vehicle_availability_matrix[f'WEEK {optimization_idx + 1}'].dropna()).tolist()

        # The temp_locations_in_final_routes makes sure we do not show the route back to the main depot
        # straight-away in the plot of this week.
        temp_locations_in_final_routes = copy.deepcopy(locations_in_final_routes)

        # Update the starting points and record what has just happened.
        starting_points = []
        broken_down_vehicles_this_week = []

        """The most important for-loop. It updates the starting points and records the travelled distance!"""
        # Index the previous functioning vehicle ids list.
        for route_index in range(len(functioning_vehicle_ids_this_week)):
            # Access the old starting position of a vehicle via index 0 in a route.
            left_location_index = routes_indices_pyvrp[route_index][0]
            # Access the new, current location of a vehicle via an index 1 in a route.
            visited_location_index = routes_indices_pyvrp[route_index][1]
            # Add the transfer drive kilometers. Note that despite using dummy clients, we use the actual distances.
            # This, we can do because the dummy clients are added at the back!
            currently_driven_total_transfer_drive_kilometers_of_the_campaign += full_distances[left_location_index][
                visited_location_index]
            # Record the visited coords for the final tour.
            locations_in_final_routes[functioning_vehicle_ids_this_week[route_index]].append(
                full_locations_list_dummy_clients_pyvrp[visited_location_index])
            temp_locations_in_final_routes[functioning_vehicle_ids_this_week[route_index]].append(
                full_locations_list_dummy_clients_pyvrp[visited_location_index])

            # Only add a visited location as a starting point if the vehicle is still part of the
            # functioning vehicle ids list OF THE NEXT WEEK!
            location_in_question = full_locations_list_dummy_clients_pyvrp[visited_location_index]
            if functioning_vehicle_ids_this_week[route_index] in functioning_vehicle_ids_next_week_theoretically:
                # Set the new starting point.
                starting_points.append(full_locations_list_dummy_clients_pyvrp[visited_location_index])
                # Remove location from the poi pool!

                """For the real past data reopt we always remove!!"""
                # Do not try to remove the home-base because it is not part of the poi pool!
                if visited_location_index != 0:
                    poi_pool.remove(location_in_question)  

            else:  
                # Vehicle broke down and is immediately sent back to the depot.
                # Add the distance of the return to the end-depot to the total transfer drive KMs.
                currently_driven_total_transfer_drive_kilometers_of_the_campaign += \
                    full_distances[visited_location_index][0]
                # Record the depot coords.
                locations_in_final_routes[functioning_vehicle_ids_this_week[route_index]].append(homebase)
                # Record the broken-down vehicle ids.
                broken_down_vehicles_this_week.append(functioning_vehicle_ids_this_week[route_index])

                """For the real past data reopt we always remove!!"""
                # Do not try to remove the home-base because it is not part of the poi pool!
                if visited_location_index != 0:
                    poi_pool.remove(location_in_question)

        # Check for new incoming functioning vehicle ids. And append the depot to the starting points list.
        new_vehicle_ids_incoming = [id_new for id_new in functioning_vehicle_ids_next_week_theoretically if
                                    id_new not in functioning_vehicle_ids_this_week]
        for idx in range(len(new_vehicle_ids_incoming)):
            starting_points.append(homebase)

        # Use a new functioning_vehicle_ids list to set up the next optimization iteration.
        # Put the incoming vehicle ids at the end.
        for id_to_remove in broken_down_vehicles_this_week:
            functioning_vehicle_ids_this_week.remove(id_to_remove)
        functioning_vehicle_ids = functioning_vehicle_ids_this_week + new_vehicle_ids_incoming

        """
        # Create a final folium map. Formerly "if final_optimization_step"
        if optimization_idx == campaign_length_in_weeks:
            m = folium.Map(name_to_coords_dict['HKS'], zoom_start=5, tiles='Cartodb Positron')

            temp_vehicle_index = 0
            for vehicle_id in all_vehicle_ids:
                locations_currently_visited_in_single_route = temp_locations_in_final_routes[vehicle_id]
                vehicle_group = folium.FeatureGroup(name=f'{vehicle_id}')

                for poi_name in locations_currently_visited_in_single_route[1:]:
                    folium.Marker(
                        icon=folium.Icon(icon="thumbtack", prefix='fa', color='gray',
                                         icon_color=all_vehicle_colors[temp_vehicle_index]),
                        location=name_to_coords_dict[poi_name],
                        tooltip=f'{poi_name}'
                    ).add_to(vehicle_group)

                # Add the homebase marker.
                folium.Marker(
                    location=name_to_coords_dict['HKS'],
                    icon=folium.Icon(icon="house", prefix='fa', color='black'),
                ).add_to(vehicle_group)

                for visited_location_idx in range(len(locations_currently_visited_in_single_route) - 1):
                    # Retrieve geo data of this tour segment.
                    geo_data_tour_segment = geo_dict[locations_currently_visited_in_single_route[visited_location_idx]][
                        locations_currently_visited_in_single_route[visited_location_idx + 1]]

                    line = {"type": "Feature",
                            "properties": {
                                "color": all_vehicle_colors[temp_vehicle_index],
                                "fillColor": all_vehicle_colors[temp_vehicle_index],
                                "stroke-width": 5},
                            "geometry": geo_data_tour_segment}

                    style_function = lambda x: {
                        'color': x['properties']['color'],
                        'weight': x['properties']['stroke-width'],
                        'fillColor': x['properties']['fillColor']
                    }
                    # Please note that for GeoJson the coordinate format is [LONG, LAT]!
                    folium.GeoJson(line, style_function=style_function,
                                   tooltip=f"{vehicle_id}").add_to(vehicle_group)

                vehicle_group.add_to(m)
                temp_vehicle_index += 1

            # Add the layer control.
            folium.LayerControl().add_to(m)
            # Make folium fullscreen.
            m.add_child(Fullscreen())
            m.save(f'Actual_drives_map.html')
        """

        # End of campaign:
        # The last time we add new locations to the poi pool after an optimization is
        # in the week/iteration No. (campaign_length_in_weeks - order_horizon)
        if optimization_idx <= campaign_length_in_weeks - order_horizon:
            # t.o.b.h. + optimization_idx - 1 to look to the plannned locations that are the t.o.b.h. ahead
            new_locations = df_planned_locations[f'WEEK {order_horizon + optimization_idx}'].dropna().tolist()
            poi_pool += new_locations

        # If the poi pool remains empty after disruptions & dynamics it means you have just visited
        # your last location and nothing bad has happened. But the vehicles still need to drive home!
        """
        if not poi_pool and not final_optimization_step:
            poi_pool = [homebase] * len(functioning_vehicle_ids)
            final_optimization_step = True
        """

        # Update the starting point indices and the number of endpoints.
        starting_point_indices = list(range(1, len(functioning_vehicle_ids) + 1))
        end_points_indices = [0] * len(functioning_vehicle_ids)

        # Update the coords list and recalculate the distance matrix.
        full_locations_list = [homebase] + starting_points + poi_pool
        full_locations_list_dummy_clients_pyvrp = ([homebase] + starting_points
                                                   + poi_pool + starting_points)

        full_distances = generate_distance_matrix(full_locations_list, dist_dict)
        full_distances_dummy_clients_pyvrp = generate_distance_matrix(full_locations_list_dummy_clients_pyvrp,
                                                                      dist_dict)

        # Update the vehicle capacities and the demands.
        vehicle_capacities = ([math.ceil(len(poi_pool) / len(functioning_vehicle_ids))]
                              * len(functioning_vehicle_ids))
        vehicle_capacities_dummy_clients_pyvrp = (
                [math.ceil((len(poi_pool) + len(starting_point_indices)) / len(functioning_vehicle_ids))]
                * len(functioning_vehicle_ids))
        full_demands = [0] + [0] * len(functioning_vehicle_ids) + [1] * len(poi_pool)
        full_demands_dummy_clients_pyvrp = ([0] + [0] * len(functioning_vehicle_ids) + [1] * len(poi_pool)
                                            + [1] * len(functioning_vehicle_ids))

        functioning_vehicle_ids_this_week = functioning_vehicle_ids.copy()

        # Set the time that the solver has to get to a solution.
        max_runtime_in_seconds = (len(poi_pool) + len(functioning_vehicle_ids_this_week))*2.08

        optimization_idx += 1
        
        # print(f"Current transfer drive kilometers: {currently_driven_total_transfer_drive_kilometers_of_the_campaign}\n")
        

    # Double-check the final distance after the while loop!
    total_driving_distance_in_km = 0
    for vehicle_id in all_vehicle_ids:
        temp_single_final_route = locations_in_final_routes[vehicle_id]
        for location_idx in range(len(temp_single_final_route) - 1):
            # Retrieve the distance between two locations of a tour.
            dist_tour_segment = dist_dict[temp_single_final_route[location_idx]][
                temp_single_final_route[location_idx + 1]]
            total_driving_distance_in_km += dist_tour_segment
    print(f'\nTotal travel distance (double check!): {total_driving_distance_in_km} km')

    """
    print(f"\nActually driven routes:")
    for vehicle_id in all_vehicle_ids:
        temp_single_final_route = locations_in_final_routes[vehicle_id]
        print(f"Actually driven route of vehicle {vehicle_id}: {temp_single_final_route}\n")
    """

if __name__ == '__main__':
    seeds = [37, 6, 24, 68, 153, 402, 87, 2]
    order_horizons = [4, 6, 8, 10, 12, 14, 16, 18, 20]
    #2 schon gemacht vom seed 37, jetzt mit oh 4 weiter!
    for seed in seeds:
        print(f"Seed: {seed}")
        # Have atleast the 42 seeds for tomorrow!!
        for order_horizon in order_horizons:
            print(f"OH: {order_horizon}")
            main(_proposed_seed=seed, _proposed_order_horizon=order_horizon)
