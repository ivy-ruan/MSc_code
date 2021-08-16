import geojson
import geopandas as gpd
import pandas as pd
import folium
import utm
from shapely import geometry
from shapely import ops
import numpy as np
import os
import warnings
warnings. filterwarnings("ignore")

def utm_convert(row):
    ls = []
    for y in row['location']:
        ls.append(list(utm.from_latlon(y[1],y[0])[0:2]))
        # convert location information in latitude and longitude to Universal Transverse Mercator coordinate system (UTM). 
        # This is especially useful for large dense arrays in a small area
    return ls 

def linestring(row):
    return geometry.LineString(row.location)

def polygon(row):
    try: 
        temp = geometry.Polygon(row.location)
        return temp
    except:
        return np.nan

def load_highway(where):
    osmRoads = []
    for i in range(1,6):
        path = f'./data/geojson/{where}/map{i}.geojson'
        with open(path, encoding="utf-8") as f:
            osmlines = geojson.load(f)
        for allFeatures in osmlines.features:
            if 'highway' in allFeatures['properties']:
                roadinfo = allFeatures['properties']
        
                locarr = np.array(allFeatures['geometry']['coordinates'], dtype="object")
        
                if locarr.ndim == 3:
                    locarr = locarr.reshape(-1, locarr.shape[-1])  # 3-dim -> 2-dim
        
                if locarr.ndim == 1:
                    locarr = np.array([locarr,locarr]) 
        
                roadinfo['location'] = locarr
        
                try:
                    osmRoads.append(roadinfo)
                except:
                    continue
        
    osmRoads = pd.DataFrame.from_dict(osmRoads)
    osmRoads = osmRoads.set_index('osm_id')

    osmRoads['utmLocation'] = osmRoads.apply(lambda row: utm_convert(row), axis=1) # convert to utm coordinate
    osmRoads['locationLineString'] = osmRoads.apply(lambda row: linestring(row), axis=1)
    osmRoads = osmRoads.filter(['name','highway', 'location', 'utmLocation', 'locationLineString'])

    for i in range(len(osmRoads)):
        road = osmRoads.iloc[i]
        
        if road['highway'] == 'motorway' or road['highway'] == 'motorway_junction' or road['highway'] == 'motorway_link':
            osmRoads.iloc[i]['highway'] = 'motorway'
        elif road['highway'] == 'primary' or road['highway'] == 'primary_link':
            osmRoads.iloc[i]['highway'] = 'primary_secondary'
        elif road['highway'] == 'secondary' or road['highway'] == 'secondary_link':
            osmRoads.iloc[i]['highway'] = 'primary_secondary'
        elif road['highway'] == 'tertiary' or road['highway'] == 'tertiary_link':
            osmRoads.iloc[i]['highway'] = 'tertiary'
        elif road['highway'] == 'residential' or road['highway'] == 'living_street' or road['highway'] == 'service':
            osmRoads.iloc[i]['highway'] = 'residential'
        elif road['highway'] == 'footway' or road['highway'] == 'cycleway' or road['highway'] == 'pedestrian' \
        or road['highway'] == 'path' or road['highway'] == 'steps':
            osmRoads.iloc[i]['highway'] = 'footway'
        elif road['highway'] == 'unclassified':
            osmRoads.iloc[i]['highway'] = 'unclassified'                                                       
        else:
            osmRoads.iloc[i]['highway'] = 'unknown'
       
    
    osm_roads = gpd.GeoDataFrame(osmRoads)
    osm_roads = osm_roads.rename({'locationLineString': 'geometry'}, axis = 1)
    osm_roads['geometry'] = gpd.GeoSeries(osm_roads['geometry'])
                    
    return osm_roads

def add_road_feature(osm_roads, data_path):
    full_data = pd.read_csv(data_path)
    points_full = gpd.GeoDataFrame(full_data, geometry = gpd.points_from_xy(full_data['gpsLongitude'],full_data['gpsLatitude']))
    offset = 0.00025 # Roughly 50 meters
    bbox_full = points_full.bounds + [-offset, -offset, offset, offset]
    hits_full = bbox_full.apply(lambda row: list(osm_roads.sindex.intersection(row)), axis=1)
    
    dist_df_full = pd.DataFrame({'pt_idx':np.repeat(hits_full.index, hits_full.apply(len)), 'close_road_idx':np.concatenate(hits_full.values)})
    dist_df_full = dist_df_full.join(points_full['geometry'].rename('point'), on='pt_idx')
    dist_df_full = dist_df_full.join(osm_roads[['geometry','location', 'highway']].reset_index(drop=True), on='close_road_idx')

    dist_gdf_full = gpd.GeoDataFrame(dist_df_full)
    dist_gdf_full['distance'] = dist_gdf_full['geometry'].distance(gpd.GeoSeries(dist_gdf_full['point']))
    dist_gdf_full = dist_gdf_full.sort_values(by=['distance'])
    dist_gdf_full = dist_gdf_full.groupby('pt_idx').first()
    new_full_data = full_data.join(dist_gdf_full[['highway', 'distance', 'close_road_idx']])
    new_full_data['highway'] = new_full_data['highway'].fillna('unknown')
    new_full_data['distance'] = new_full_data['distance'].fillna(0.005)
    
    return new_full_data

def load_landuse(where):
    osmLands = []
    for i in range(1,6):
        path = f'./data/geojson/{where}/map{i}.geojson' 
        with open(path, encoding="utf-8") as f:
            osmlines = geojson.load(f)
        for allFeatures in osmlines.features:
            if 'landuse' in allFeatures['properties']:
                ## OUTLAND
                if allFeatures["properties"]["landuse"] == "grass" or allFeatures["properties"]["landuse"] == "basin" \
                or allFeatures["properties"]["landuse"] == "forest" or allFeatures["properties"]["landuse"] == "greenfield" \
                or allFeatures["properties"]["landuse"] == "meadow" or allFeatures["properties"]["landuse"] == "orchard" \
      or allFeatures["properties"]["landuse"] == "plant_nursery" or allFeatures["properties"]["landuse"] == "recreation_ground"\
      or allFeatures["properties"]["landuse"] == "village_green" or allFeatures["properties"]["landuse"] == "wasteland"\
      or allFeatures["properties"]["landuse"] == "farmland" or allFeatures["properties"]["landuse"] == "farmyard":
                    landinfo = allFeatures['properties']
                    landinfo["landuse"] = 'outland'
                    locarr = np.array(allFeatures['geometry']['coordinates'], dtype="object")
                    if locarr.ndim > 2:
                        locarr = locarr.reshape(-1, locarr.shape[-1])  # 3-dim -> 2-dim        
                    if locarr.ndim == 1:
                        locarr = np.array([locarr,locarr])        
                    if locarr.shape[0] > 2:
                        landinfo['location'] = locarr
                        osmLands.append(landinfo)
            
                ## BUILDING
                elif allFeatures["properties"]["landuse"] == "commercial" or allFeatures["properties"]["landuse"] == "retail":
                    landinfo = allFeatures['properties']
                    landinfo["landuse"] = 'building'
                    locarr = np.array(allFeatures['geometry']['coordinates'], dtype="object")
                    if locarr.ndim > 2:
                        locarr = locarr.reshape(-1, locarr.shape[-1])  # 3-dim -> 2-dim        
                    if locarr.ndim == 1:
                        locarr = np.array([locarr,locarr])        
                    if locarr.shape[0] > 2:
                        landinfo['location'] = locarr
                        osmLands.append(landinfo)
   
                ## RESIDENTIAL        
                elif allFeatures["properties"]["landuse"] == "residential":
                    landinfo = allFeatures['properties']
                    locarr = np.array(allFeatures['geometry']['coordinates'], dtype="object")
                    if locarr.ndim > 2:
                        locarr = locarr.reshape(-1, locarr.shape[-1])  # 3-dim -> 2-dim        
                    if locarr.ndim == 1:
                        locarr = np.array([locarr,locarr])        
                    if locarr.shape[0] > 2:
                        landinfo['location'] = locarr
                        osmLands.append(landinfo) 
        
            if "leisure" in allFeatures['properties']:
                ## OUTLAND(PARK)
                if allFeatures["properties"]["leisure"] == "park" or allFeatures["properties"]["leisure"] == "garden"\
                or allFeatures["properties"]["leisure"] == "pitch" or allFeatures["properties"]["leisure"] == "playground"\
                or allFeatures["properties"]["leisure"] == "recreation_ground":
                    landinfo = allFeatures['properties']
                    landinfo["landuse"] = 'outland'
                    locarr = np.array(allFeatures['geometry']['coordinates'], dtype="object")
                    if locarr.ndim > 2:
                        locarr = locarr.reshape(-1, locarr.shape[-1])  # 3-dim -> 2-dim        
                    if locarr.ndim == 1:
                        locarr = np.array([locarr,locarr])        
                    if locarr.shape[0] > 2:
                        landinfo['location'] = locarr
                        osmLands.append(landinfo)
                
            if "building" in allFeatures["properties"]:
                ## RESIDENTIAL
                if allFeatures["properties"]["building"] == "house" or allFeatures["properties"]["building"] == "apartments" \
            or allFeatures["properties"]["building"] == "residential" or allFeatures["properties"]["building"] == "detached"\
            or allFeatures["properties"]["building"] == "hotel":
                    landinfo = allFeatures['properties']
                    landinfo["landuse"] = 'residential'
                    locarr = np.array(allFeatures['geometry']['coordinates'], dtype="object")
                    if locarr.ndim > 2:
                        locarr = locarr.reshape(-1, locarr.shape[-1])  # 3-dim -> 2-dim        
                    if locarr.ndim == 1:
                        locarr = np.array([locarr,locarr])        
                    if locarr.shape[0] > 2:
                        landinfo['location'] = locarr
                        osmLands.append(landinfo)
            
                ## BUILDING
                elif allFeatures["properties"]["building"] == "yes" or allFeatures["properties"]["building"] == "university" \
            or allFeatures["properties"]["building"] == "church" or allFeatures["properties"]["building"] == "office" \
            or allFeatures["properties"]["building"] == "museum" or allFeatures["properties"]["building"] == "school"\
            or allFeatures["properties"]["building"] == "commercial" or allFeatures["properties"]["building"] == "retail":
                    landinfo = allFeatures['properties']
                    landinfo["landuse"] = 'building'
                    locarr = np.array(allFeatures['geometry']['coordinates'], dtype="object")
                    if locarr.ndim > 2:
                        locarr = locarr.reshape(-1, locarr.shape[-1])  # 3-dim -> 2-dim        
                    if locarr.ndim == 1:
                        locarr = np.array([locarr,locarr])        
                    if locarr.shape[0] > 2:
                        landinfo['location'] = locarr
                        osmLands.append(landinfo)
                    
    osmLands = pd.DataFrame.from_dict(osmLands)
    osmLands = osmLands.set_index('osm_id')
    
    # Filtering and pre-processing 
    # osmLands = osmLands[osmLands['type'] != 'multipolygon']
    #osmLands['utmLocation'] = osmLands.apply(lambda row: utm_convert(row), axis=1) # convert to utm coordinate
    osmLands['locationPolygon'] = osmLands.apply(lambda row: polygon(row), axis=1)
    # construct the line using a list of coordinate-tuples
    osmLands = osmLands.filter(['name','landuse', 'location', 'locationPolygon'])
    # only care about related attributes 
    osmLands.dropna(subset=['locationPolygon'], inplace=True)
    osm_lands = gpd.GeoDataFrame(osmLands)
    osm_lands = osm_lands.rename({'locationPolygon': 'geometry'}, axis = 1)
    osm_lands['geometry'] = gpd.GeoSeries(osm_lands['geometry'])
    return osm_lands

def add_land_feature(osm_lands, data):
    full_data = data
    points_full = gpd.GeoDataFrame(full_data, geometry = gpd.points_from_xy(full_data['gpsLongitude'],full_data['gpsLatitude']))
    offset = 0.00025 # Roughly 50 meters
    bbox_full = points_full.bounds + [-offset, -offset, offset, offset]
    hits_full = bbox_full.apply(lambda row: list(osm_lands.sindex.intersection(row)), axis=1)
    
    dist_df_full = pd.DataFrame({'pt_idx':np.repeat(hits_full.index, hits_full.apply(len)), 'close_land_idx': np.concatenate(hits_full.values)})
    dist_df_full = dist_df_full.join(full_data['geometry'].rename('point'), on='pt_idx')
    dist_df_full = dist_df_full.join(osm_lands[['geometry','location', 'landuse']].reset_index(drop=True), on='close_land_idx')

    dist_gdf_full = gpd.GeoDataFrame(dist_df_full)
    dist_gdf_full['if_contain'] = dist_gdf_full['geometry'].contains(gpd.GeoSeries(dist_gdf_full['point']))
    
    contain = dist_gdf_full[dist_gdf_full['if_contain'] == True]
    df_contain = contain[['pt_idx', 'landuse', 'close_land_idx']].set_index('pt_idx')
    new_full_data = full_data.join(df_contain)
    new_full_data['landuse'].fillna('unknown', inplace=True)
    
    return new_full_data

def distance_euclidean(data):
    '''
    Calculate the Euclidean distance between two GPS points based on the longitude and latitude
    :param data: DataFrame --> Needs to include gpsLongitude and gpsLatitude features
    :return: DataFrame with gps_distance included as additional feature
    '''

    N = data.shape[0]
    gps_dist = [0] * N
    data['gps_dist'] = gps_dist

    for n in range(1, N):
        data['gps_dist'].iloc[n] = np.sqrt((data['gpsLongitude'].iloc[n-1] - data['gpsLongitude'].iloc[n])**2 \
                                               + (data['gpsLatitude'].iloc[n-1] - data['gpsLatitude'].iloc[n])**2)

    return data

def calculate_std(data, column_name, k=10):
    '''
    Calculates the standard deviation of a given columns
    :param data: DataFrame --> data
    :param column_name: String --> Column name for the standard deviation is to be calculated
    :param k: Int --> Window size
    :return: DataFrame --> Data with additional column for the standard deviation (column_name_std)
    '''
    N = data.shape[0]
    var = [0] * N
    data[column_name + '_std'] = var

    n = k

    while n < N:
        data[column_name + '_std'].iloc[n] = np.std(data[column_name].iloc[n-k:n])
        n += 1
    return data

def view_missing_value(df):
    for column in list(df.columns):
        print("{}:  {} % missing values \n".format(column, ((len(df) - df[column].count()) / len(df))*100))
        
def min_max_norm(data, col):
    target_col = data[col]
    max_num = max(target_col.dropna())
    min_num = min(target_col.dropna())
    std = (target_col - min_num) / (max_num - min_num)
    data[col] = std
    
    return data
        

