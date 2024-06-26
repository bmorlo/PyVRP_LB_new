"""Sprintplanungs-Matrixerweiterungs-tool // V3 // 18.05.2024 // Benedikt Morlock"""
import math as mt
import pandas as pds
import json
import openrouteservice as ors
from openrouteservice import convert
from time import sleep


def main(_file_name, _api_key):
    # Read latitude and longitude data from excel.
    df_input = pds.read_excel(f'{_file_name}.xlsx', sheet_name='Location_names_to_coords')
    all_location_names = df_input['NAME'].tolist()

    # Generate the coords as a list of lists. [[LONG, LAT], ..., [LONG, LAT]]. The API needs (1) LAT (2) LONG.
    name_to_coords_dict = {}
    for idx in range(0, len(all_location_names)):
        name_to_coords_dict[all_location_names[idx]] = [df_input['LONG'].tolist()[idx], df_input['LAT'].tolist()[idx]]

    # Generate a list of location names that are new aka they are not yet used as keys in the .json files/dictionaries.
    with open(
            f'BM_dev/Main_code/json_files_for_dist_and_geo_matrices/revised_distance_matrix_row_{all_location_names[0]}.json') as dismarow:
        dist_row_as_dict = json.load(dismarow)
    old_location_names = list(dist_row_as_dict[all_location_names[0]])
    new_locations_to_append = [location for location in all_location_names if location not in old_location_names]

    outer_ITER_IDX = 0
    api_request_count = 0
    # Iterate over all existing location .json-files (distance and geo data) and
    # append the distance and geo datas FROM an old TO a new location.
    for ITER_IDX in range(113, len(all_location_names) - len(new_locations_to_append)):
        # Call the ors-API and get the distance and the route geometry.

        distances_to_append = []
        geo_datas_to_append = []
        for new_location in new_locations_to_append:
            # Request distance and geo data from the API TO the new locations.

            # Avoid running into the per-minute token limit.
            if api_request_count >= 35:
                api_request_count = 0
                sleep(70)

            route_ors = ors.client.directions(
                client=ors.Client(key=_api_key,
                                  requests_kwargs={'verify': False}),
                coordinates=[name_to_coords_dict[all_location_names[ITER_IDX]], name_to_coords_dict[new_location]],
                profile='driving-car',
                validate=False
            )
            # Note that the distance returned by the ors-API is in meters --> /1000 to get to km!
            dist_of_route = route_ors['routes'][0]['segments'][0]['distance']
            distances_to_append.append(mt.ceil(dist_of_route / 1000))
            # Get the decoded route geometry.
            decoded_route = convert.decode_polyline(route_ors['routes'][0]['geometry'])
            # Only save start + every tenth element + end coordinate.
            decoded_route['coordinates'] = ([decoded_route['coordinates'][0]] + decoded_route['coordinates'][1:-1:10] +
                                            [decoded_route['coordinates'][-1]])
            geo_datas_to_append.append(decoded_route)
            api_request_count += 1

        # Makes sure we can start where we left off and create the missing
        # .json-files / FROM matrix rows for the new locations.
        outer_ITER_IDX = ITER_IDX

        # Only rework the .json files if we actually have something to rework!
        if new_locations_to_append:
            # Read in the old files.
            with open(
                    f'BM_dev/Main_code/json_files_for_dist_and_geo_matrices/revised_distance_matrix_row_{all_location_names[ITER_IDX]}.json') as dismarow:
                dist_row_as_dict = json.load(dismarow)
            with open(
                    f'BM_dev/Main_code/json_files_for_dist_and_geo_matrices/revised_geometry_matrix_row_{all_location_names[ITER_IDX]}.json') as geomarow:
                geo_row_as_dict = json.load(geomarow)
            # Append the new data.
            for new_location_index in range(len(new_locations_to_append)):
                dist_row_as_dict[all_location_names[ITER_IDX]][new_locations_to_append[new_location_index]] = (
                    distances_to_append)[new_location_index]
                geo_row_as_dict[all_location_names[ITER_IDX]][new_locations_to_append[new_location_index]] = (
                    geo_datas_to_append)[new_location_index]
            # Save the new files.
            with open(f'BM_dev/Main_code/revised_distance_matrix_row_{all_location_names[ITER_IDX]}.json', 'x') as dismarow:
                json.dump(dist_row_as_dict, dismarow)
            with open(f'BM_dev/Main_code/revised_geometry_matrix_row_{all_location_names[ITER_IDX]}.json', 'x') as geomarow:
                json.dump(geo_row_as_dict, geomarow)

    # Generate the new distance and geo data matrix rows with the new locations as FROM nodes.
    api_request_count = 0
    for final_ITER_IDX in range(outer_ITER_IDX + 1, len(all_location_names)):
        # Start at the first new location as a FROM node.
        # Call the ors-API and get the distance and the route geometry.
        dist_matrix_final_row = {}
        geo_matrix_final_row = {}
        temp_row_of_dist_matrix = {}
        temp_row_of_geo_matrix = {}

        # Get distance and geo data TO all the locations.
        for to_location_name in all_location_names:
            # Avoid running into the per-minute token limit.
            if api_request_count >= 35:
                api_request_count = 0
                sleep(70)

            route_ors = ors.client.directions(
                client=ors.Client(key=_api_key,
                                  requests_kwargs={'verify': False}),
                coordinates=[name_to_coords_dict[all_location_names[final_ITER_IDX]],
                             name_to_coords_dict[to_location_name]],
                profile='driving-car',
                validate=False
            )
            # Note that the distance returned by the ors-API is in meters --> /1000 to get to km!
            dist_of_route = route_ors['routes'][0]['segments'][0]['distance']
            temp_row_of_dist_matrix[to_location_name] = mt.ceil(dist_of_route / 1000)
            # Get the decoded route geometry.
            decoded_route = convert.decode_polyline(route_ors['routes'][0]['geometry'])
            # Only save start + every tenth element + end coordinate.
            decoded_route['coordinates'] = ([decoded_route['coordinates'][0]] + decoded_route['coordinates'][1:-1:10] +
                                            [decoded_route['coordinates'][-1]])
            temp_row_of_geo_matrix[to_location_name] = decoded_route
            api_request_count += 1

        dist_matrix_final_row[all_location_names[final_ITER_IDX]] = temp_row_of_dist_matrix
        geo_matrix_final_row[all_location_names[final_ITER_IDX]] = temp_row_of_geo_matrix

        with open(f'BM_dev/Main_code/revised_distance_matrix_row_{all_location_names[final_ITER_IDX]}.json', 'x') as dismarow:
            json.dump(dist_matrix_final_row, dismarow)
        with open(f'BM_dev/Main_code/revised_geometry_matrix_row_{all_location_names[final_ITER_IDX]}.json', 'x') as geomarow:
            json.dump(geo_matrix_final_row, geomarow)


if __name__ == '__main__':
    """
    (1) Please make sure to remove any remaining .json files in the main project folder of PyCharm.
    (2) Please also check that the folder 'json_files_for_dist_and_geo_matrices' that is located in the
        main project folder of PyCharm has the latest .json files before starting the application.
    (2) Please also specify the name of the Excel input file. Do not change the sheet names within the Excel file!
    (3) If the API returned an error (i.e., the program exited with something else than a 'code 0'), 
        simply delete the in this failed attempt so far generated .json files in the main project folder of PyCharm
        and rerun the application once more.

    (4) Please specify the openrouteservice API key as a string.
    (5) Possible API keys registered under Benedikt Morlock are:
        # 2000 Tokens: '5b3ce3597851110001cf62482931a1ef940e44cb9ddcd7111ead4c26'
        # 10000 Tokens: '5b3ce3597851110001cf624865baa215e227495497ee3e00c87e4f39'
    (6) Please note that the token limit of an API key refreshes completely after 24h after the FIRST token was used up.
    """
    main(_file_name='BM_dev/Main_code/Sprint_Planungs_Tool_Input_Data_Europa',
         _api_key='5b3ce3597851110001cf624865baa215e227495497ee3e00c87e4f39')
