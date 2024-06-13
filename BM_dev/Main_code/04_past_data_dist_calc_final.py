import pandas as pds
import json
import folium
from folium.plugins import Fullscreen
import matplotlib as mplb
import numpy as np

"""We are using symmetric distances to recalculate the actually driven tours."""

# Read city names and latitude and longitude data from the Excel.
df_location_names_to_coords = pds.read_excel('BM_dev/Main_code/Central_Input_Data_Andre.xlsx', 'Location_names_to_coords')
df_vehicle_ids = pds.read_excel('BM_dev/Main_code/Central_Input_Data_Andre.xlsx', sheet_name='Vehicle_ids')
df_past_tours = pds.read_excel('BM_dev/Main_code/Central_Input_Data_Andre.xlsx', sheet_name='Past_tours')
df_planned_locations = pds.read_excel('BM_dev/Main_code/Central_Input_Data_Andre.xlsx', 'Planned_locations')
all_location_names = df_location_names_to_coords['NAME'].tolist()

# Create a dictionary that maps a name to the coordinates.
temp_row_index_location_name_to_coords = 0
location_name_to_coords_dict = {}
for location in df_location_names_to_coords['NAME']:
    location_name_to_coords_dict[location] = [
        df_location_names_to_coords['LAT'][temp_row_index_location_name_to_coords].item(),
        df_location_names_to_coords['LONG'][temp_row_index_location_name_to_coords].item()]
    temp_row_index_location_name_to_coords += 1

dist_dict = {}
geo_dict = {}

# Add the dist and geo rows as a sub-dict to their respective outer dicts.
for ITER_IDX in range(len(all_location_names)):
    with (open(f'BM_dev/Main_code/json_files_for_dist_and_geo_matrices/revised_distance_matrix_row_{all_location_names[ITER_IDX]}.json')
          as dismarow):
        dist_row = json.load(dismarow)
    dist_dict.update(dist_row)
    with open(f'BM_dev/Main_code/json_files_for_dist_and_geo_matrices/revised_geometry_matrix_row_{all_location_names[ITER_IDX]}.json') as geomarow:
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

# Create the map. In folium we need to change the order again of long-lat to lat-long.
m = folium.Map(location=[df_location_names_to_coords['LAT'][0], df_location_names_to_coords['LONG'][0]], zoom_start=5, tiles='Cartodb Positron')
# Generate the colors:
color_vector = np.linspace(0.1, 0.9, 16)
cmap = mplb.colormaps['turbo']
all_vehicle_colors = []
for color_float in color_vector:
    color_rgba = cmap(color_float)
    color_hex = mplb.colors.to_hex(color_rgba, keep_alpha=True)
    all_vehicle_colors.append(color_hex)

total_km_of_past_data = 0
# We have 16 tours/vehicles.
for tour_idx in range(16):
    # We have 20 weeks (excluding the distance back to the depot)
    temp_vehicle_id = df_vehicle_ids['ID'][tour_idx]
    vehicle_group = folium.FeatureGroup(name=f'{temp_vehicle_id}')

    # Add the visited locations of a tour to the map.
    tour_row = df_past_tours.values.tolist()[tour_idx]
    for poi_name in tour_row[1:]:
        folium.Marker(
            icon=folium.Icon(icon="thumbtack", prefix='fa', color='gray', icon_color=f'{all_vehicle_colors[tour_idx]}'),
            location=location_name_to_coords_dict[poi_name],
            tooltip=f'{poi_name}'
        ).add_to(vehicle_group)

    # Add the homebase.
    folium.Marker(
        location=[df_location_names_to_coords['LAT'][0], df_location_names_to_coords['LONG'][0]],
        icon=folium.Icon(icon="house", prefix='fa', color='black', icon_color='white'),
        tooltip=all_location_names[0]
    ).add_to(vehicle_group)

    for visited_location_idx in range(20):
        # Retrieve the distance between two locations of a tour.
        dist_tour_segment = dist_dict[df_past_tours[f"WEEK {visited_location_idx}"][tour_idx]][
            df_past_tours[f"WEEK {visited_location_idx + 1}"][tour_idx]]
        total_km_of_past_data += dist_tour_segment

        # Retrieve geo data of this tour segment.
        geo_data_tour_segment = geo_dict[df_past_tours[f"WEEK {visited_location_idx}"][tour_idx]][
            df_past_tours[f"WEEK {visited_location_idx + 1}"][tour_idx]]
        # TODO: ADD THIS LINE TO GET STRAIGHT-LINE DISTANCES!
        # geo_data_tour_segment['coordinates'] = [geo_data_tour_segment['coordinates'][0], geo_data_tour_segment['coordinates'][-1]]
        line = {"type": "Feature",
                "properties": {"color": all_vehicle_colors[tour_idx], "fillColor": all_vehicle_colors[tour_idx],
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
                       tooltip=f"{df_vehicle_ids['ID'][tour_idx]}").add_to(vehicle_group)

    # Add the feature group to the map
    vehicle_group.add_to(m)
# Add the layer control.
folium.LayerControl().add_to(m)
# Make folium fullscreen.
m.add_child(Fullscreen())

# m.save('past_data_map_straight_linesV4.html')
m.save('past_data_map_real_linesV4.html')
print(total_km_of_past_data)
