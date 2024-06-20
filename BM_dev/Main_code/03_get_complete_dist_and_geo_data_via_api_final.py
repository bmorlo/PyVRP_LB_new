import math as mt
import pandas as pds
import json
import openrouteservice as ors
from openrouteservice import convert
from time import sleep


def main(_api_key):
    # Read latitude and longitude data from excel.
    df_input = pds.read_excel('Sprint_Planungs_Tool_Input_Data_Europa.xlsx', sheet_name='Location_names_to_coords')
    all_location_names = df_input['NAME'].tolist()

    # Generate the coords as a list of lists. [[LONG, LAT], ..., [LONG, LAT]]
    # Note that the ors API takes [LONG, LAT] coordinate format and returns its polyline also in [LONG, LAT] format.
    coords = []
    for row_idx in range(0, len(all_location_names)):
        coords.append([df_input['LONG'][row_idx], df_input['LAT'][row_idx]])

    api_request_count = 0
    for ITER_IDX in range(107, len(all_location_names)):

        # Call the ors-API and get the distance and the route geometry.
        # Use dicts of dicts (rows) that consist of dict entries (columns). We use names.
        dist_matrix_final_row = {}
        geo_matrix_final_row = {}
        temp_row_of_dist_matrix = {}
        temp_row_of_geo_matrix = {}

        col_index = 0
        for to_coord in coords:
            # Don't run
            if api_request_count >= 35:
                api_request_count = 0
                sleep(90)

            route_ors = ors.client.directions(
                client=ors.Client(key=_api_key,
                                  requests_kwargs={'verify': False}),
                coordinates=[coords[ITER_IDX], to_coord],
                profile='driving-car',
                validate=False
            )

            # Note that the distance returned by the ors-API is in meters --> /1000 to get to km!
            temp_dist_ors = mt.ceil((route_ors['routes'][0]['segments'][0]['distance']) / 1000)
            temp_row_of_dist_matrix[all_location_names[col_index]] = temp_dist_ors
            # Get the decoded route geometry.
            decoded_route = convert.decode_polyline(route_ors['routes'][0]['geometry'])
            # Only save start + every tenth element + end coordinate.
            decoded_route['coordinates'] = ([decoded_route['coordinates'][0]] + decoded_route['coordinates'][1:-1:10] +
                                            [decoded_route['coordinates'][-1]])
            temp_row_of_geo_matrix[all_location_names[col_index]] = decoded_route
            col_index += 1

            api_request_count += 1

        dist_matrix_final_row[all_location_names[ITER_IDX]] = temp_row_of_dist_matrix
        geo_matrix_final_row[all_location_names[ITER_IDX]] = temp_row_of_geo_matrix

        with open(f'revised_distance_matrix_row_{all_location_names[ITER_IDX]}.json', 'x') as dismarow:
            json.dump(dist_matrix_final_row, dismarow)
        with open(f'revised_geometry_matrix_row_{all_location_names[ITER_IDX]}.json', 'x') as geomarow:
            json.dump(geo_matrix_final_row, geomarow)


if __name__ == '__main__':
    """
    (1) Please specify the openrouteservice API key as a string.
    (2) Possible API keys registered under Benedikt Morlock are:
        # 2000 Tokens: '5b3ce3597851110001cf62482931a1ef940e44cb9ddcd7111ead4c26'
        # 10000 Tokens: '5b3ce3597851110001cf624865baa215e227495497ee3e00c87e4f39'
    (3) Please note that the token limit of an API key refreshes completely after 24h after the FIRST token was used up.
    """
    main(_api_key='5b3ce3597851110001cf624865baa215e227495497ee3e00c87e4f39')
