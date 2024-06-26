"""Sprintplanungs-VRP-tool // V2 // 07.05.2024 // Benedikt Morlock"""
import math
import json

import numpy as np
import pandas as pds

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


def read_inputs(_excel_input_filename):
    """Reads the inputs from the central Excel file."""
    df_vehicle_ids = pds.read_excel(f'{_excel_input_filename}.xlsx', sheet_name='Starting_locations')['ID']
    df_locations_and_coords = pds.read_excel(f'{_excel_input_filename}.xlsx', sheet_name='Location_names_to_coords')
    df_planned_locations = pds.read_excel(f'{_excel_input_filename}.xlsx', sheet_name='Planned_locations')
    df_starting_points = pds.read_excel(f'{_excel_input_filename}.xlsx', sheet_name='Starting_locations')['START']
    all_location_names = df_locations_and_coords['NAME'].tolist()

    # Create a dictionary that maps a location name to its coordinates.
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
        with open(
                f'BM_dev/Main_code/json_files_for_dist_and_geo_matrices/revised_distance_matrix_row_{all_location_names[ITER_IDX]}.json') as dismarow:
            dist_row = json.load(dismarow)
        dist_dict.update(dist_row)
        with open(
                f'BM_dev/Main_code/json_files_for_dist_and_geo_matrices/revised_geometry_matrix_row_{all_location_names[ITER_IDX]}.json') as geomarow:
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

    return (df_vehicle_ids, df_locations_and_coords, df_planned_locations, dist_dict,
            geo_dict, location_name_to_coords_dict, df_starting_points)


def generate_distance_matrix(_locations, _dist_dict):
    """Returns a list of distance matrix rows aka the distance matrix belonging to the current problem instance."""
    distance_matrix = []
    for i in range(len(_locations)):
        frm_name = _locations[i]
        row_i_of_distance_matrix = []
        for j in range(len(_locations)):
            to_name = _locations[j]
            # Find the correct distance
            dist_tour_segment = _dist_dict[frm_name][to_name]
            row_i_of_distance_matrix.append(dist_tour_segment)
        distance_matrix.append(row_i_of_distance_matrix)

    return distance_matrix


def generate_optimized_tours_with_pyvrp(_max_runtime_in_seconds,
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
        node_indices_all_tours_current_optimization_pyvrp.append([start] + clients)

    # print(f"Objective value PyVRP: {result_pyvrp.cost()}")
    # The latter returns the vehicle ids for double-checking purposes.
    return node_indices_all_tours_current_optimization_pyvrp, [str(vehicle_type) for vehicle_type in
                                                               pyvrp_model.vehicle_types]


def main(_excel_input_file_name, _homebase_name_as_string, _seed, _manual_capacity_override_if_more_vehicles_than_tickets=0):
    """Run the optimizations for a specific order horizon length"""

    # Read in the central inputs.
    (df_vehicle_ids, df_locations_and_coords, df_planned_locations,
     dist_dict, geo_dict, name_to_coords_dict, df_starting_points) = read_inputs(_excel_input_file_name)

    # Define the problem instance.
    order_horizon = df_planned_locations.shape[1]
    vehicle_ids = df_vehicle_ids.tolist()

    # Depending on its vehicle ID, set the color of a tour.
    color_vector = np.linspace(0.1, 0.9, len(vehicle_ids))
    cmap = mplb.colormaps['turbo']
    all_vehicle_colors = []
    for color_float in color_vector:
        color_rgba = cmap(color_float)
        color_hex = mplb.colors.to_hex(color_rgba, keep_alpha=True)
        all_vehicle_colors.append(color_hex)
    vehicle_id_to_color_dictionary = dict(zip(vehicle_ids, all_vehicle_colors))

    # Initialize start and endpoints.
    starting_points = df_starting_points.tolist()
    starting_point_indices = list(range(1, len(starting_points) + 1))
    end_points_indices = [0] * len(vehicle_ids)

    # Depending on how many tickets we have, build up the poi pool.
    poi_pool = df_planned_locations['WEEK 1'].dropna().tolist()
    for week_num in range(2, order_horizon + 1):
        poi_pool += df_planned_locations[f'WEEK {week_num}'].dropna().tolist()

    # Initialize the full location list.
    homebase = _homebase_name_as_string
    full_poi_list_dummy_clients_pyvrp = [homebase] + starting_points + poi_pool + starting_points
    full_distances_dummy_clients_pyvrp = generate_distance_matrix(full_poi_list_dummy_clients_pyvrp, dist_dict)

    # Initialize the vehicle capacities.
    if _manual_capacity_override_if_more_vehicles_than_tickets != 0:
        vehicle_capacities_dummy_clients_pyvrp = [_manual_capacity_override_if_more_vehicles_than_tickets 
                                                  + 1] * len(vehicle_ids)
    else:
        vehicle_capacities_dummy_clients_pyvrp = ([math.ceil((len(poi_pool) + len(vehicle_ids)) / len(
            vehicle_ids))] * len(vehicle_ids))

    # Create demands that correspond to 1) the end depot, 2) the starting points,
    # 3) the points of interest [and 4) the dummy clients].
    full_demands_dummy_clients_pyvrp = ([0] + [0] * len(vehicle_ids) + [1] * len(poi_pool)
                                        + [1] * len(vehicle_ids))

    expected_driving_distance_in_km = 0

    # Set the time that the solver has to get to a solution.
    max_runtime_in_seconds = (len(poi_pool) + len(vehicle_ids))*2.08

    # Solve the VRP using the PyVRP solver.
    full_coords_list_dummy_clients_pyvrp = [name_to_coords_dict[name] for name in
                                            full_poi_list_dummy_clients_pyvrp]
    full_coords_list_dummy_clients_pyvrp = [[math.ceil(lat), math.ceil(long)] for lat, long in
                                            full_coords_list_dummy_clients_pyvrp]
    tours_indices_pyvrp, details_of_pyvrp_vehicle_types = generate_optimized_tours_with_pyvrp(
        max_runtime_in_seconds, full_coords_list_dummy_clients_pyvrp, full_distances_dummy_clients_pyvrp,
        vehicle_capacities_dummy_clients_pyvrp, full_demands_dummy_clients_pyvrp, starting_point_indices,
        end_points_indices, vehicle_ids, _seed)

    # Save and print the tours via a dictionary.
    planned_tours = {}
    for tour_index in range(len(vehicle_ids)):
        planned_tours[vehicle_ids[tour_index]] = [full_poi_list_dummy_clients_pyvrp[loc_idx]
                                                  for loc_idx in tours_indices_pyvrp[tour_index]]
        temp_planned_tour = planned_tours[vehicle_ids[tour_index]]
        print(f"Planned tour of vehicle {vehicle_ids[tour_index]}: {temp_planned_tour}\n")

        # Calculate the distance of a tour.
        temp_planned_tour_distance = 0
        for poi_idx in range(len(temp_planned_tour) - 1):
            # Retrieve the distance between two locations of a tour.
            dist_tour_segment = dist_dict[temp_planned_tour[poi_idx]][temp_planned_tour[poi_idx + 1]]
            temp_planned_tour_distance += dist_tour_segment
        # Add the tour distance to the total expected driving distance.
        expected_driving_distance_in_km += temp_planned_tour_distance

    # Create the map of the tour planning.
    m = folium.Map(name_to_coords_dict[homebase], zoom_start=5, tiles="Cartodb Positron")

    temp_vehicle_index = 0
    for vehicle_id in vehicle_ids:
        locations_to_be_visited_in_vehicle_tour = planned_tours[vehicle_id]
        vehicle_group = folium.FeatureGroup(name=f'{vehicle_id}')

        # Add the visited locations of a tour to the map.
        for poi_name in locations_to_be_visited_in_vehicle_tour[1:]:
            folium.Marker(
                icon=folium.Icon(icon="thumbtack", prefix='fa', color='gray',
                                 icon_color=f'{vehicle_id_to_color_dictionary[vehicle_id]}'),
                location=name_to_coords_dict[poi_name],
                tooltip=f'{poi_name}'
            ).add_to(vehicle_group)

        # Add the starting points to the map.
        folium.Marker(
            location=name_to_coords_dict[starting_points[temp_vehicle_index]],
            icon=folium.Icon(icon="car-side", prefix='fa', color='black',
                             icon_color=f'{vehicle_id_to_color_dictionary[vehicle_id]}'),
            tooltip=f'{starting_points[temp_vehicle_index]}: Tour start of vehicle {vehicle_id}'
        ).add_to(vehicle_group)

        for visited_location_idx in range(len(locations_to_be_visited_in_vehicle_tour) - 1):
            # Retrieve geo data of this tour segment.
            geo_data_tour_segment = geo_dict[locations_to_be_visited_in_vehicle_tour[visited_location_idx]][
                locations_to_be_visited_in_vehicle_tour[visited_location_idx + 1]]

            line = {"type": "Feature",
                    "properties": {
                        "color": vehicle_id_to_color_dictionary[vehicle_id],
                        "fillColor": vehicle_id_to_color_dictionary[vehicle_id],
                        "stroke-width": 5},
                    "geometry": geo_data_tour_segment
                    }

            style_function = lambda x: {
                'color': x['properties']['color'],
                'weight': x['properties']['stroke-width'],
                'fillColor': x['properties']['fillColor']
            }
            # Please note that for GeoJson the coordinate format is [LONG, LAT]!
            folium.GeoJson(line, style_function=style_function,
                           tooltip=f"Planned tour of vehicle {vehicle_id}: "
                                   f"{locations_to_be_visited_in_vehicle_tour}").add_to(vehicle_group)

        # Add the feature group to the map
        vehicle_group.add_to(m)
        temp_vehicle_index += 1

    # Add the layer control.
    folium.LayerControl().add_to(m)
    # Make folium fullscreen.
    m.add_child(Fullscreen())
    # Save the map.
    m.save(f'Planned_tours_real.html')
    # Print the expected KMs.
    print(f"Expected driving distance: {expected_driving_distance_in_km} km\n")


if __name__ == '__main__':
    """
    (1) Please specify the name of the Excel input file. Do not change the sheet names within the Excel file!
    (2) Please specify the name of the homebase for plotting purposes.
    (3) Please note that the program is internally working with symmetric distance and geodata matrices.
    (4) Please also make sure to specify a "manual_capacity_override_if_more_vehicles_than_tickets" greater than zero
        if more vehicles than tickets exist!
    """    
    # capas = [5, 6, 7, 8, 9, 10]
    # for capa in capas:
        # print(f"Manuelle Kapazität: {capa} Tickets")
    main(_excel_input_file_name='BM_dev/Main_code/Sprint_Planungs_Tool_Input_Data_Europa_2_FZG', 
            _homebase_name_as_string='HKS', _seed=42, _manual_capacity_override_if_more_vehicles_than_tickets=0)
        
