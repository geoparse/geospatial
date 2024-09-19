import collections
import json
import math
import re
from datetime import datetime, timedelta
from math import atan2, cos, radians, sin, sqrt
from multiprocessing import Pool, cpu_count
from time import time

import folium  # Folium is a Python library used for visualising geospatial data. Actually, it's a Python wrapper
import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
import requests  # used in geocoding()
from folium import plugins  # for Leaflet which is a leading open-source JavaScript library for plotting interactive maps.
from h3 import h3
from polygon_geohasher.polygon_geohasher import geohash_to_polygon, polygon_to_geohashes
from s2 import s2

# from keplergl import KeplerGl    # keplergl visuslises data on the map
from shapely.geometry import (  # mapping converts Shapely object to a dict object (GeoJson)
    LineString,
    MultiPolygon,
    Point,
    Polygon,
)
from shapely.ops import transform  # shape converts a dict object (GeoJson) to Shapely object

pd.options.mode.chained_assignment = None  # default='warn'


def color_map(col, head=None, tail=None):  # col: column name, head and tail: substring indices
    palettet = [  # colour palette
        "#e6194b",  # red
        "#4363d8",  # blue
        "#3cb44b",  # green
        "#800000",  # maroon (dark red)
        "#008080",  # teal (dark green)
        "#000080",  # navy (dark blue)
        "#f58231",  # orange
        "#911eb4",  # purple
        "#808000",  # olive
        "#9a6324",  # brown
        "#f032e6",  # magenta
        "#dfb119",  # dark yellow
        "#42d4f4",  # cyan
        "#808080",  # grey
        "#e12348",  # Bright Red
        "#dc2c46",  # Strong Red
        "#d73644",  # Vivid Red
        "#cd4a40",  # Deep Red
        "#c8543e",  # Intense Red
        "#c25e3c",  # Fire Red
        "#bd683a",  # Scarlet
        "#b77238",  # Fiery Orange
        "#b27c36",  # Tangerine
        "#ad8634",  # Burnt Orange
        #        '#a79032', # Rust Orange
        #        '#a29a30', # Pumpkin
        #        '#9da42e', # Goldenrod
        #        '#98ae2c', # Saffron
        #        '#93b82a', # Amber
        #        '#8ec228', # Apricot
        #        '#89cc26', # Peach
        #        '#84d624', # Cantaloupe
        #        '#7fde22', # Honeydew
        #        '#7ae820', # Lime
        #        '#75f21e', # Chartreuse
        #        '#70fc1c', # Neon Green
        #        '#6bff1c', # Fluorescent Green
        #        '#6bff49', # Grass Green
        #        '#6bff83', # Periwinkle
        #        '#6bffbc', # Pink
    ]

    if isinstance(col, int):
        idx = col % len(palettet)  # palette index
    else:
        col = str(col)  # convert to string
        col = re.sub(r"[\W_]+", "", col)  # remove characters which are not numbers and alphabets using re
        idx = int(col[head:tail], 36) % len(palettet)  # convert text to integer (10 digits + 26 letters = 36)
    return palettet[idx]


def base_map(sw, ne):
    karta = folium.Map(tiles=None)
    tiles = {
        "cartodbpositron": "Bright Mode",
        "cartodbdark_matter": "Dark Mode",
    }
    for item in tiles:
        folium.TileLayer(item, name=tiles[item], max_zoom=21).add_to(karta)

    folium.TileLayer(
        name="Satellite",
        attr="Esri",
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        overlay=False,
        control=True,
        max_zoom=19,
    ).add_to(karta)

    folium.TileLayer("openstreetmap", name="OpenStreetMap", max_zoom=19).add_to(karta)

    karta.fit_bounds([sw, ne])
    return karta


def geom_to_cell(
    geoms, cell_type, res, dump=False
):  # s2.polyfill() function covers the hole in a polygon too (which is not correct).
    polys = []  # geom_to_cell_parallel() function splits a polygon to smaller polygons without holes
    for geom in geoms:
        if geom.geom_type == "Polygon":
            polys += [geom.__geo_interface__]
        elif geom.geom_type == "MultiPolygon":  # if multipolygon
            polys += [g.__geo_interface__ for g in geom.geoms]  # a list of dicts (polygons)

    cells = []
    if cell_type == "geohash":
        cells = set()
        for geom in geoms:
            cells |= polygon_to_geohashes(geom, precision=res, inner=False)
        cells = list(cells)

    elif cell_type == "s2":
        for poly in polys:
            cells += s2.polyfill(
                poly, res, geo_json_conformant=True, with_id=True
            )  # returns json object with id and geometry (coords)
        cells = [item["id"] for item in cells]  # remove geometry and keep id only
        cells = list(set(cells))  # remove duplicates

    elif cell_type == "h3":
        for poly in polys:
            cells += h3.polyfill(poly, res, geo_json_conformant=True)

    if dump:
        with open(f"~/Desktop/{cell_type}/{datetime.now()}.txt", "w") as json_file:
            json.dump(cells, json_file)
        return
    else:
        return cells


def geom_to_cell_parallel(mdf, cell_type, res, compact=False, verbose=False):
    if verbose:
        print(datetime.now())
        print("Slicing the bbox of mdf ... ", end="")
        start_time = time()
    n_cores = cpu_count()
    slices = 128 * n_cores

    minlon, minlat, maxlon, maxlat = mdf.total_bounds
    dlon = maxlon - minlon
    dlat = maxlat - minlat
    ratio = dlon / dlat

    x_cells = round(sqrt(slices) * ratio)
    y_cells = round(sqrt(slices) / ratio)

    steplon = dlon / x_cells
    steplat = dlat / y_cells

    grid_polygons = []
    for lat in np.arange(minlat, maxlat, steplat):  # Iterate over the rows and columns to create the grid
        for lon in np.arange(minlon, maxlon, steplon):  # Calculate the coordinates of the current grid cell
            llon, llat, ulon, ulat = (lon, lat, lon + steplon, lat + steplat)  # lower lat, upper lat
            polygon = Polygon([(llon, llat), (ulon, llat), (ulon, ulat), (llon, ulat)])
            grid_polygons.append(polygon)  # Add the polygon to the list
    gmdf = gpd.GeoDataFrame(geometry=grid_polygons, crs=mdf.crs)  # Create a GeoDataFrame for the grid polygons

    if verbose:
        elapsed_time = round(time() - start_time)
        print(f"{elapsed_time} seconds.   {slices} slices")
        start_time = time()
        print("Performing the intersection between the gridded bbox and mdf ... ", end="")
    gmdf = gpd.overlay(mdf, gmdf, how="intersection")  # grid mdf

    if verbose:
        elapsed_time = round(time() - start_time)
        print(f"{elapsed_time} seconds.   {len(gmdf)} slices")
        start_time = time()
        print("Calculating the cells for all geometries of the gridded mdf in parallel ... ", end="")
    gmdf = gmdf.sample(
        frac=1
    )  # Shuffle the rows of gmdf to have a good balance of small (incomplete squares) and big (full squares) in all chunks
    geom_chunks = np.array_split(list(gmdf.geometry), 4 * n_cores)
    inputs = zip(geom_chunks, [cell_type] * 4 * n_cores, [res] * 4 * n_cores)

    # Create a multiprocessing pool and apply the overlay function in parallel on each chunk
    with Pool(n_cores) as pool:
        cells = pool.starmap(geom_to_cell, inputs)
    cells = [item for sublist in cells for item in sublist]  # flatten the list

    if cell_type == "geohash":  # do nothing if cell_type = h3
        if verbose:
            elapsed_time = round(time() - start_time)
            print(f"{elapsed_time} seconds")
            start_time = time()
            print("Removing duplicate cells ... ", end="")
        cells = list(set(cells))  # remove duplicated cells

    elif cell_type == "s2":
        if verbose:
            elapsed_time = round(time() - start_time)
            print(f"{elapsed_time} seconds")
            start_time = time()
            print("Removing duplicate cells ... ", end="")
        cells = list(set(cells))  # remove duplicates

    cell_counts = len(cells)

    if compact:
        if verbose:
            elapsed_time = round(time() - start_time)
            print(f"{elapsed_time} seconds")
            start_time = time()
            print("Compacting cells ... ", end="")
        cells = compact_cells(cells, cell_type)
        if verbose:
            elapsed_time = round(time() - start_time)
            print(f"{elapsed_time} seconds")
    return cells, cell_counts


def geom_to_cell_parallel_2(mdf, cell_type, res, compact=False, verbose=False, dump=True):
    if verbose:
        print(datetime.now())
        print("Slicing the bbox of mdf ... ", end="")
        start_time = time()
    n_cores = cpu_count()
    slices = 128 * n_cores

    minlon, minlat, maxlon, maxlat = mdf.total_bounds
    dlon = maxlon - minlon
    dlat = maxlat - minlat
    ratio = dlon / dlat

    x_cells = round(sqrt(slices) * ratio)
    y_cells = round(sqrt(slices) / ratio)

    steplon = dlon / x_cells
    steplat = dlat / y_cells

    grid_polygons = []
    for lat in np.arange(minlat, maxlat, steplat):  # Iterate over the rows and columns to create the grid
        for lon in np.arange(minlon, maxlon, steplon):  # Calculate the coordinates of the current grid cell
            llon, llat, ulon, ulat = (lon, lat, lon + steplon, lat + steplat)  # lower lat, upper lat
            polygon = Polygon([(llon, llat), (ulon, llat), (ulon, ulat), (llon, ulat)])
            grid_polygons.append(polygon)  # Add the polygon to the list
    gmdf = gpd.GeoDataFrame(geometry=grid_polygons, crs=mdf.crs)  # Create a GeoDataFrame for the grid polygons

    if verbose:
        elapsed_time = round(time() - start_time)
        print(f"{elapsed_time} seconds.   {slices} slices")
        start_time = time()
        print("Performing the intersection between the gridded bbox and mdf ... ", end="")
    gmdf = gpd.overlay(mdf, gmdf, how="intersection")  # grid mdf

    if verbose:
        elapsed_time = round(time() - start_time)
        print(f"{elapsed_time} seconds.   {len(gmdf)} slices")
        start_time = time()
        print("Calculating the cells for all geometries of the gridded mdf in parallel ... ", end="")
    gmdf = gmdf.sample(
        frac=1
    )  # Shuffle the rows of gmdf to have a good balance of small (incomplete squares) and big (full squares) in all chunks
    geom_chunks = np.array_split(list(gmdf.geometry), 4 * n_cores)
    inputs = zip(geom_chunks, [cell_type] * 4 * n_cores, [res] * 4 * n_cores, [dump] * 4 * n_cores)

    # Create a multiprocessing pool and apply the overlay function in parallel on each chunk
    with Pool(n_cores) as pool:
        pool.starmap(geom_to_cell, inputs)
    return


def cell_to_geom(cells, cell_type):
    res = [
        len(cell)
        if cell_type == "geohash"
        else cell[1]
        if cell_type == "h3"
        else s2.CellId.from_token(cell).level()  # cell = token
        for cell in cells
    ]

    geoms = [
        geohash_to_polygon(cell)
        if cell_type == "geohash"
        else Polygon(s2.s2_to_geo_boundary(cell, geo_json_conformant=True))
        if cell_type == "s2"
        else Polygon(h3.h3_to_geo_boundary(cell, geo_json=True))
        for cell in cells
    ]

    return res, geoms


def compact_cells(cells, cell_type):
    if cell_type == "h3":
        return list(h3.compact(cells))
    elif cell_type == "s2":
        #    cells = [item['id'] for item in cells]
        cells = [s2.CellId.from_token(item) for item in cells]
        res = cells[0].level()  # assuming all s2 cells have the same resolution (the output of geom_to_cell() function)
        num_children = 4
    elif cell_type == "geohash":
        res = len(cells[0])
        num_children = 32

    compact_cells = []
    for i in range(res, 0, -1):
        parent_ids = [cell.parent() if cell_type == "s2" else cell[: i - 1] for cell in cells]
        count_dict = collections.Counter(parent_ids)
        idx = [
            i for i, item in enumerate(parent_ids) if count_dict.get(item, 0) == num_children
        ]  # get indices of items with count 4 in parent_ids

        mask = [True] * len(cells)
        for ix in idx:
            mask[ix] = False
        cells = [item for i, item in enumerate(cells) if mask[i]]

        compact_cells += cells
        cells = list(set([item for item in parent_ids if count_dict.get(item, 0) == num_children]))
    compact_cells += cells

    if cell_type == "geohash":
        return compact_cells
    else:  # s2
        return [item.to_token() for item in compact_cells]


def uncompact_s2(compact_tokens, level):
    uncompact_tokens = []
    for token in compact_tokens:
        cell_id = s2.CellId.from_token(token)
        uncompact_tokens += list(cell_id.children(level))
    uncompact_tokens = [item.to_token() for item in uncompact_tokens]
    return list(set(uncompact_tokens))


def points(
    row, karta, color, color_head=None, color_tail=None, opacity=0.5, radius=3, weight=6, popup_dict=None, x=None, y=None
):  # color_head, color_tail: color substring indices
    try:
        location = [row.geometry.y, row.geometry.x]
    except Exception:
        location = [row[y], row[x]]  # x, y: lon, lat column names in DataFrame

    if color in row.index:  # color in df.columns
        color = color_map(row[color], color_head, color_tail)

    if popup_dict is None:
        popup = None
    else:
        popup = ""
        for item in popup_dict:
            popup += "{}: <b>{}</b><br>".format(item, row[popup_dict[item]])

    folium.CircleMarker(location=location, radius=radius, color=color, opacity=opacity, weight=weight, tooltip=popup).add_to(
        karta
    )
    return 0


def row_polygons(row, karta, fill_color, line_width, popup_dict):
    if fill_color in row.index:
        fill_color = color_map(row[fill_color])

    def style_function(x):
        return {"fillColor": fill_color, "color": "#000000", "fillOpacity": 0.25, "weight": line_width}

    def highlight_function(x):
        return {"fillColor": fill_color, "color": "#000000", "fillOpacity": 0.5, "weight": line_width}

    if popup_dict is None:
        popup = None
    else:
        popup = ""
        for item in popup_dict:
            popup += "<b>{}</b>: <b>{}</b><br>".format(item, row[popup_dict[item]])

    gjson = row.geometry.__geo_interface__
    gjson = folium.GeoJson(data=gjson, style_function=style_function, highlight_function=highlight_function, tooltip=popup)
    gjson.add_to(karta)
    return 0


def polygons(karta, mdf, fill_color, highlight_color, line_width, popup_dict):
    if fill_color in mdf.columns:
        fill_color = color_map(mdf[fill_color].values[0])

    def style_function(x):
        return {"fillColor": fill_color, "color": "#000000", "fillOpacity": 0.25, "weight": line_width}

    def highlight_function(x):
        return {"fillColor": highlight_color, "color": "#000000", "fillOpacity": 0.5, "weight": line_width}

    for _, row in mdf.iterrows():
        if popup_dict is None:
            popup = None
        else:
            popup = ""
            for item in popup_dict:
                popup += "<b>{}</b>: <b>{}</b><br>".format(item, row[popup_dict[item]])

        gjson = gpd.GeoSeries(row["geometry"]).to_json()
        gjson = folium.GeoJson(data=gjson, style_function=style_function, highlight_function=highlight_function, tooltip=popup)
        gjson.add_to(karta)
    return 0


def plp(  # plp: points, lines, polygons
    gdf_list=None,
    # Point
    x=None,
    y=None,  # provide x and y if more than one column in gdf contains 'lat' and 'lon'
    cluster=False,
    heatmap=False,
    line=False,
    antpath=False,
    point_color="blue",
    color_head=None,
    color_tail=None,  # color_head and color_tail: substring indices
    point_opacity=0.5,
    point_radius=3,
    point_weight=6,
    point_popup=None,  # point_weight = 2xpoint_radius
    buffer_radius=0,
    ring_inner_radius=0,
    ring_outer_radius=0,
    # LineString
    line_color="blue",
    line_opacity=0.5,
    line_weight=6,
    line_popup=None,
    # Polygon
    centroid=False,  # if centroid=True it shows centroids of polygons on the map.
    fill_color="red",
    highlight_color="green",
    line_width=0.3,
    polygon_popup=None,
    choropleth_cols=None,
    choropleth_bins=None,
    choropleth_legend=None,
    choropleth_palette="YlOrRd",
    choropleth_highlight=True,
    geohash_res=0,
    s2_res=-1,
    h3_res=-1,
    geohash_inner=False,
    compact=False,
    cells=None,
    cell_type=None,  # list of geohash, S2 or H3 cell IDs
    ways=None,  # list of OSM way IDs (lines or polygons) and Overpass API URL to query from
    url=None,
):
    if cells:
        res, geoms = cell_to_geom(cells, cell_type=cell_type)
        gdf = gpd.GeoDataFrame({"id": cells, "res": res, "geometry": geoms}, crs="EPSG:4326")
        karta = plp(gdf, polygon_popup={"ID": "id", "Resolution": "res"})
        return karta

    if ways:
        geoms = ways2geom(ways, url)
        gdf = gpd.GeoDataFrame({"way_id": ways, "geometry": geoms}, crs="EPSG:4326")
        if isinstance(gdf.geometry[0], LineString):
            karta = plp(gdf, line_popup={"way_id": "way_id"}, line_color="way_id")
        else:
            karta = plp(gdf, polygon_popup={"way_id": "way_id"}, fill_color="way_id")
        return karta

    if isinstance(gdf_list, pd.DataFrame):
        gdf_list = [gdf_list]

    minlat, maxlat, minlon, maxlon = 90, -90, 180, -180
    for gdf in gdf_list:
        if not isinstance(gdf, gpd.GeoDataFrame):
            if not x:  # if x=None (x and y are not specified)
                xx = [col for col in gdf.columns if "lon" in col.lower() or "lng" in col.lower()][0]
                yy = [col for col in gdf.columns if "lat" in col.lower()][0]
            lons = gdf[xx]
            lats = gdf[yy]
            minlatg, minlong, maxlatg, maxlong = min(lats), min(lons), max(lats), max(lons)  # minlatg: minlat in gdf
        else:
            minlong, minlatg, maxlong, maxlatg = gdf.total_bounds
        minlat, minlon = min(minlat, minlatg), min(minlon, minlong)
        maxlat, maxlon = max(maxlat, maxlatg), max(maxlon, maxlong)

    sw = [minlat, minlon]  # South West (bottom left)
    ne = [maxlat, maxlon]  # North East (top right)
    karta = base_map(sw, ne)

    i = 0  # index of gdf in gdf_list
    for gdf in gdf_list:
        i += 1
        if not isinstance(gdf, gpd.GeoDataFrame):  # if pd.DataFrame
            geom = None
        else:
            geom = gdf.geometry.values[0]

        if isinstance(geom, Polygon) or isinstance(geom, MultiPolygon):
            if centroid:
                group_centroid = folium.FeatureGroup(name=f"{i}- Centroid")
                cdf = gpd.GeoDataFrame({"geometry": gdf.centroid}, crs="EPSG:4326")  # centroid df
                cdf.apply(points, karta=group_centroid, color="red", axis=1)
                group_centroid.add_to(karta)
            if choropleth_cols:
                group_chor = folium.FeatureGroup(name=f"{i}- Choropleth")
                choropleth_plp(
                    karta=group_chor,
                    gdf=gdf,
                    columns=choropleth_cols,
                    bins=choropleth_bins,
                    legend=choropleth_legend,
                    palette=choropleth_palette,
                    highlight=choropleth_highlight,
                )
                group_chor.add_to(karta)
            else:
                group_polygon = folium.FeatureGroup(name=f"{i}- Polygon")
                gdf.apply(
                    row_polygons,
                    karta=group_polygon,
                    fill_color=fill_color,
                    line_width=line_width,
                    popup_dict=polygon_popup,
                    axis=1,
                )
                group_polygon.add_to(karta)

        if isinstance(geom, LineString):
            group_line = folium.FeatureGroup(name=f"{i}- Line")
            for _index, row in gdf.iterrows():
                coordinates = [
                    (coord[1], coord[0]) for coord in row.geometry.coords
                ]  # Convert LineString geometries to coordinates (lat, lon)

                if line_color in gdf.columns:
                    color = color_map(row[line_color])
                else:
                    color = line_color

                if line_popup is None:
                    popup = None
                else:
                    popup = ""
                    for item in line_popup:
                        popup += "{}: <b>{}</b><br>".format(item, row[line_popup[item]])

                group_line.add_child(
                    folium.PolyLine(coordinates, color=color, weight=line_weight, opacity=line_opacity, tooltip=popup)
                )
            group_line.add_to(karta)

        if not isinstance(gdf, gpd.GeoDataFrame) or isinstance(geom, Point):
            if not isinstance(gdf, gpd.GeoDataFrame) and not x:  # if x=None (x and y are not specified)
                xx = [col for col in gdf.columns if "lon" in col.lower() or "lng" in col.lower()][0]
                yy = [col for col in gdf.columns if "lat" in col.lower()][0]
            else:
                xx = yy = None

            group_point = folium.FeatureGroup(name=f"{i}- Point")
            gdf.apply(
                points,
                x=xx,
                y=yy,
                karta=group_point,
                color=point_color,
                color_head=color_head,
                color_tail=color_tail,
                opacity=point_opacity,
                radius=point_radius,
                weight=point_weight,
                popup_dict=point_popup,
                axis=1,
            )
            group_point.add_to(karta)

            if cluster:
                group_cluster = folium.FeatureGroup(name=f"{i}- Cluster")
                if not isinstance(gdf, gpd.GeoDataFrame):
                    group_cluster.add_child(plugins.MarkerCluster(locations=list(zip(lats, lons))))
                else:
                    group_cluster.add_child(plugins.MarkerCluster(locations=list(zip(gdf.geometry.y, gdf.geometry.x))))
                group_cluster.add_to(karta)

            if heatmap:
                group_heatmap = folium.FeatureGroup(name=f"{i}- Heatmap")
                if not isinstance(gdf, gpd.GeoDataFrame):
                    group_heatmap.add_child(plugins.HeatMap(list(zip(lats, lons)), radius=10))
                else:
                    group_heatmap.add_child(plugins.HeatMap(list(zip(gdf.geometry.y, gdf.geometry.x)), radius=10))
                group_heatmap.add_to(karta)

            if line:
                group_line = folium.FeatureGroup(name=f"{i}- Line")
                if not isinstance(gdf, gpd.GeoDataFrame):
                    group_line.add_child(
                        folium.PolyLine(list(zip(lats, lons)), color=line_color, weight=line_weight, opacity=line_opacity)
                    )
                else:
                    group_line.add_child(
                        folium.PolyLine(
                            list(zip(gdf.geometry.y, gdf.geometry.x)),
                            color=line_color,
                            weight=line_weight,
                            opacity=line_opacity,
                        )
                    )
                group_line.add_to(karta)

            if antpath:
                group_antpath = folium.FeatureGroup(name=f"{i}- Ant Path")
                if not isinstance(gdf, gpd.GeoDataFrame):
                    group_antpath.add_child(plugins.AntPath(list(zip(lats, lons))))
                else:
                    group_antpath.add_child(plugins.AntPath(list(zip(gdf.geometry.y, gdf.geometry.x))))
                group_antpath.add_to(karta)

            if buffer_radius > 0:
                group_buffer = folium.FeatureGroup(name=f"{i}- Buffer")
                bgdf = gdf.copy()  # buffered gdf
                bgdf["geometry"] = (
                    bgdf.to_crs(find_proj(gdf.geometry.values[0])).buffer(buffer_radius).to_crs("EPSG:4326")
                )  # radius in meters
                polygons(
                    karta=group_buffer,
                    mdf=bgdf,
                    fill_color=fill_color,
                    highlight_color=fill_color,
                    line_width=line_width,
                    popup_dict=None,
                )
                group_buffer.add_to(karta)

            if ring_outer_radius > 0:
                group_ring = folium.FeatureGroup(name=f"{i}- Ring")
                bgdf = gdf.copy()  # buffered gdf
                bgdf["geometry"] = (
                    bgdf.to_crs(find_proj(gdf.geometry.values[0]))
                    .buffer(ring_outer_radius)
                    .difference(bgdf.to_crs(find_proj(gdf.geometry.values[0])).buffer(ring_inner_radius))
                    .to_crs("EPSG:4326")
                )  # radius in meters
                polygons(
                    karta=group_ring,
                    mdf=bgdf,
                    fill_color=fill_color,
                    highlight_color=fill_color,
                    line_width=line_width,
                    popup_dict=None,
                )
                group_ring.add_to(karta)

    if geohash_res > 0:  # inner=False doesn't work if compact=True
        if isinstance(geom, Polygon) or isinstance(geom, MultiPolygon):
            cdf = gdf.copy()
        else:
            bb = Polygon([[minlon, minlat], [maxlon, minlat], [maxlon, maxlat], [minlon, maxlat], [minlon, minlat]])
            cdf = gpd.GeoDataFrame({"geometry": [bb]}, crs="EPSG:4326")  # cell df

        cells, _ = geom_to_cell_parallel(cdf, cell_type="geohash", res=geohash_res, compact=compact)
        # ids, res, geoms = cell_to_geom(cells, cell_type='geohash')
        res, geoms = cell_to_geom(cells, cell_type="geohash")
        cdf = gpd.GeoDataFrame({"id": cells, "res": res, "geometry": geoms}, crs="EPSG:4326")

        group_geohash = folium.FeatureGroup(name="Geohash")
        polygons(
            karta=group_geohash,
            mdf=cdf,
            fill_color=fill_color,
            highlight_color=highlight_color,
            line_width=line_width,
            popup_dict={"ID": "id", "Resolution": "res"},
        )
        group_geohash.add_to(karta)

    if s2_res > -1:
        if isinstance(geom, Polygon) or isinstance(geom, MultiPolygon):
            cdf = gdf.copy()
        else:
            bb = Polygon([[minlon, minlat], [maxlon, minlat], [maxlon, maxlat], [minlon, maxlat], [minlon, minlat]])
            cdf = gpd.GeoDataFrame({"geometry": [bb]}, crs="EPSG:4326")  # cell df

        cells, _ = geom_to_cell_parallel(cdf, cell_type="s2", res=s2_res, compact=compact)
        res, geoms = cell_to_geom(cells, cell_type="s2")
        cdf = gpd.GeoDataFrame({"id": cells, "res": res, "geometry": geoms}, crs="EPSG:4326")

        group_s2 = folium.FeatureGroup(name="Google S2")
        polygons(
            karta=group_s2,
            mdf=cdf,
            fill_color=fill_color,
            highlight_color=highlight_color,
            line_width=line_width,
            popup_dict={"ID": "id", "Resolution": "res"},
        )
        group_s2.add_to(karta)

    if h3_res > -1:
        if isinstance(geom, Polygon) or isinstance(geom, MultiPolygon):
            cdf = gdf.copy()
        else:
            bb = Polygon([[minlon, minlat], [maxlon, minlat], [maxlon, maxlat], [minlon, maxlat], [minlon, minlat]])
            cdf = gpd.GeoDataFrame({"geometry": [bb]}, crs="EPSG:4326")  # cell df

        cells, _ = geom_to_cell_parallel(cdf, cell_type="h3", res=h3_res, compact=compact)
        res, geoms = cell_to_geom(cells, cell_type="h3")
        cdf = gpd.GeoDataFrame({"id": cells, "res": res, "geometry": geoms}, crs="EPSG:4326")

        group_h3 = folium.FeatureGroup(name="Uber H3")
        polygons(
            karta=group_h3,
            mdf=cdf,
            fill_color=fill_color,
            highlight_color=highlight_color,
            line_width=line_width,
            popup_dict={"ID": "id", "Resolution": "res"},
        )
        group_h3.add_to(karta)
    folium.LayerControl(collapsed=False).add_to(karta)
    return karta


def choropleth_plp(karta, gdf, columns, bins, legend, palette, highlight):  # used in plp function only
    choropleth = folium.Choropleth(
        geo_data=gdf,  # containing multypolygon data
        name="Choropleth",
        data=gdf,  # containing data to be shown on map e.g. name, counts, ...
        columns=columns,
        key_on="feature.properties." + columns[0],
        legend_name=legend,  # description of columns[1]
        bins=bins,
        fill_color=palette,
        fill_opacity=0.5,
        line_opacity=0.25,
        smooth_factor=0,
        highlight=highlight,
    ).geojson.add_to(karta)
    folium.features.GeoJsonTooltip(fields=columns).add_to(choropleth)
    return 0


def choropleth(gdf, columns, bins, legend, palette="YlOrRd", highlight=True):
    minlon, minlat, maxlon, maxlat = gdf.total_bounds
    sw = [minlat, minlon]
    ne = [maxlat, maxlon]
    karta = base_map(sw, ne)

    choropleth = folium.Choropleth(
        geo_data=gdf,  # containing multypolygon data
        name="Choropleth",
        data=gdf,  # containing data to be shown on map e.g. name, counts, ...
        columns=columns,
        key_on="feature.properties." + columns[0],
        legend_name=legend,  # description of columns[1]
        bins=bins,
        fill_color=palette,
        fill_opacity=0.5,
        line_opacity=0.25,
        smooth_factor=0,
        highlight=highlight,
    ).add_to(karta)
    folium.features.GeoJsonTooltip(fields=columns).add_to(choropleth.geojson)
    folium.LayerControl(collapsed=False).add_to(karta)
    return karta


def intersection(gdf1, gdf2, poly_id=None):
    #    """
    #    Returns a subset of gdf1 intersecting with any object in gdf2.
    #    The function adds a new column to gdf2 showing the number of intersecting
    #    objects for all geometries in gdf2. It also adds the geometry id (poly_id in gdf2)
    #    to the intersected gdf.
    #    """
    int_gdf = pd.DataFrame()  # intersected gdf which is a subset of gdf1
    counts = []
    for geom in gdf2.geometry:
        gdf = gdf1[gdf1.intersects(geom)]  # intersected points
        if poly_id is not None and len(gdf) > 0:
            gid = gdf2[gdf2.geometry == geom][poly_id].iloc[0]  # gid: geometry id
            gdf["geom_id"] = gid
        int_gdf = pd.concat([int_gdf, gdf])  # selected pdf
        counts.append(len(gdf))
    gdf2["counts"] = counts
    return int_gdf


def quick_intersection(gdf1, gdf2, poly_id=None):
    int_gdf = pd.DataFrame()  # intersected gdf which is a subset of gdf1
    counts = []
    for geom in gdf2.geometry:
        # Get the indices of the objects that are likely to be inside the bounding box of the given Polygon
        pos_idx = list(gdf1.sindex.intersection(geom.bounds))
        pos_gdf = gdf1.iloc[pos_idx]  # possible gdf
        pre_gdf = pos_gdf[pos_gdf.intersects(geom)]  # precise gdf
        if poly_id is not None and len(pre_gdf) > 0:
            gid = gdf2[gdf2.geometry == geom][poly_id].iloc[0]  # gid: geometry id
            pre_gdf["geom_id"] = gid
        int_gdf = pd.concat([int_gdf, pre_gdf])  # intersected gdf
        counts.append(len(pre_gdf))
    gdf2["counts"] = counts
    return int_gdf


def flatten_3d(geom):
    """
    Takes a GeoSeries of 3D Multi/Polygons (has_z) and returns a list of 2D Multi/Polygons
    """
    new_geo = []
    for p in geom:
        if p.has_z:
            if p.geom_type == "Polygon":
                lines = [xy[:2] for xy in list(p.exterior.coords)]
                new_p = Polygon(lines)
                new_geo.append(new_p)
            elif p.geom_type == "MultiPolygon":
                new_multi_p = []
                for ap in p:
                    lines = [xy[:2] for xy in list(ap.exterior.coords)]
                    new_p = Polygon(lines)
                    new_multi_p.append(new_p)
                new_geo.append(MultiPolygon(new_multi_p))
    return new_geo


def drange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n)


def find_proj(geom):  # find projection (the UTM zone)
    """
    The Universal Transverse Mercator (UTM) is a map projection system
    for assigning coordinates to locations on the surface of the Earth.
    """
    if geom.geom_type != "Point":
        geom = geom.centroid
    lat = geom.y
    lon = geom.x
    if lat >= 0:
        proj = "EPSG:326"
    else:
        proj = "EPSG:327"
    utm = math.ceil(30 + lon / 6)
    return proj + str(utm)


def trans_proj(geom, proj1, proj2):  # transform projection
    project = pyproj.Transformer.from_crs(pyproj.CRS(proj1), pyproj.CRS(proj2), always_xy=True).transform
    return transform(project, geom)


def geom_stats(geom=None, unit="m"):
    if not geom:  # print help if geom is None
        print(
            "mdf[['nshells', 'nholes', 'nshell_points',  'area', 'border']] = [gsp.geom_stats(geom, unit='km') for geom in mdf.geometry]"
        )
        return

    utm_zone = find_proj(geom)

    if geom.geom_type == "Polygon":
        polylist = [geom]
    elif geom.geom_type == "MultiPolygon":
        polylist = list(geom.geoms)

    n_shells = len(polylist)
    n_holes = n_shell_points = border = area = 0
    for poly in polylist:
        n_holes += len(poly.interiors)
        n_shell_points += len(poly.exterior.coords)
        border += trans_proj(poly, "EPSG:4326", utm_zone).exterior.length
        area += trans_proj(poly, "EPSG:4326", utm_zone).area
    if unit == "m":  # unit in meters
        return [n_shells, n_holes, n_shell_points, round(area), round(border)]
    else:  # unit in km
        return [n_shells, n_holes, n_shell_points, round(area / 1000000), round(border / 1000)]


def google_geocoding(address_or_zipcode, api_key):
    """
    This function takes a text-based address or zip code and
    returns geographic coordinates, latitude/longitude pair,
    to identify a location on the Earth's surface.

    df[['lat', 'lon']] = df.apply(lambda row : gsp.geocoding(row.address), axis=1)
    """

    lat, lon = None, None
    base_url = "https://maps.googleapis.com/maps/api/geocode/json"
    endpoint = f"{base_url}?address={address_or_zipcode}&key={api_key}"
    r = requests.get(endpoint)
    if r.status_code not in range(200, 299):
        return None, None
    try:
        """
        This try block incase any of our inputs are invalid. This is done instead
        of actually writing out handlers for all kinds of responses.
        """
        results = r.json()["results"][0]
        lat = results["geometry"]["location"]["lat"]
        lon = results["geometry"]["location"]["lng"]
    except Exception:
        pass
    return pd.Series([lat, lon])


def google_reverse_geocoding(lat, lon, api_key):
    """
    This function takes a coordinate (lat, lon) and returns the postcode.

    df['postcode'] = df.apply(lambda row : gsp.geocoding(row.lat, row.lon), axis=1)
    """
    lat = (
        0 if abs(lat) < 0.0001 else lat
    )  # otherwise it returns the following error: "Invalid request. Invalid 'latlng' parameter."
    lon = 0 if abs(lon) < 0.0001 else lon
    # Make the reverse geocoding request
    url = f"https://maps.googleapis.com/maps/api/geocode/json?latlng={lat},{lon}&key={api_key}"
    response = requests.get(url)
    data = response.json()

    # Parse the response to extract the postcode
    if "results" in data and len(data["results"]) > 0:
        for item in data["results"]:
            for component in item.get("address_components", []):
                if (
                    "postal_code" in component.get("types", []) and len(component["types"]) == 1
                ):  # 'types': ['postal_code', 'postal_code_prefix']}],
                    return component.get("long_name", None)
    return None


def haversine(lat1, lon1, lat2, lon2):
    """
    https://en.wikipedia.org/wiki/Haversine_formula
    The haversine formula determines the great-circle distance between two points on a sphere given their longitudes and latitudes.

    https://en.wikipedia.org/wiki/Longitude
    a = 6378137.0        # equatorial radius
    b = 6356752.3142     # polar radius
    R = (2*a+b)/3        # mean radius = 6371008.7714
    """

    r = 6371008.8
    lat1, lat2, dlon = radians(lat1), radians(lat2), radians(lon2 - lon1)
    dlat = lat2 - lat1

    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))  # distance in gradians
    return r * c  # distance in meters


def vincenty(lat1, lon1, lat2, lon2):
    """
    Calculates the geodesic distance between two points on the Earth's surface
    using the Vincenty formula, which accounts for the Earth's ellipsoidal shape.

    Parameters:
    - lat1, lon1: Latitude and longitude of the first point (in degrees).
    - lat2, lon2: Latitude and longitude of the second point (in degrees).

    Returns:
    - Distance between the two points in meters.

    Notes:
    - This implementation may encounter numerical issues, such as divide-by-zero errors,
      in edge cases where the points are on opposite sides of the Earth or on the same meridian
      e.g., from (0,0) to (0,90).However, for points (0,0) to (0.001,90), the distance calculation
      is accurate within a small error margin (about 9.3e-06 meters).

    - The error in the above approach can be significant for very small distances,
      such as between (0,0) and (0,0.001).
    """

    # Constants for WGS-84 ellipsoid
    a = 6378137.0  # Equatorial radius in meters
    f = 1 / 298.257223563  # Flattening
    b = a * (1 - f)  # Polar radius

    # Convert degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    # Differences in longitude
    ll = lon2 - lon1

    # Iterative Vincenty formula
    u1 = math.atan((1 - f) * math.tan(lat1))
    u2 = math.atan((1 - f) * math.tan(lat2))
    sin_u1 = math.sin(u1)
    cos_u1 = math.cos(u1)
    sin_u2 = math.sin(u2)
    cos_u2 = math.cos(u2)

    lambda_ = ll
    lambda_prev = 0
    max_iterations = 1000
    tolerance = 1e-12

    for _ in range(max_iterations):
        sin_lambda = math.sin(lambda_)
        cos_lambda = math.cos(lambda_)
        sin_sigma = math.sqrt((cos_u2 * sin_lambda) ** 2 + (cos_u1 * sin_u2 - sin_u1 * cos_u2 * cos_lambda) ** 2)
        cos_sigma = sin_u1 * sin_u2 + cos_u1 * cos_u2 * cos_lambda
        sigma = math.atan2(sin_sigma, cos_sigma)
        sin_alpha = cos_u1 * cos_u2 * sin_lambda / sin_sigma
        cos2_alpha = 1 - sin_alpha**2
        cos2_sigma_m = cos_sigma - 2 * sin_u1 * sin_u2 / cos2_alpha
        cc = f / 16 * cos2_alpha * (4 + f * (4 - 3 * cos2_alpha))
        lambda_prev = lambda_
        lambda_ = ll + (1 - cc) * f * sin_alpha * (
            sigma + cc * sin_sigma * (cos2_sigma_m + cc * cos_sigma * (-1 + 2 * cos2_sigma_m**2))
        )

        if abs(lambda_ - lambda_prev) < tolerance:
            break
    else:
        raise ValueError("Vincenty formula did not converge")

    u2 = cos2_alpha * (a**2 - b**2) / (b**2)
    aa = 1 + u2 / 16384 * (4096 + u2 * (-768 + u2 * (320 - 175 * u2)))
    bb = u2 / 1024 * (256 + u2 * (-128 + u2 * (74 - 47 * u2)))
    delta_sigma = (
        bb
        * sin_sigma
        * (
            cos2_sigma_m
            + bb
            / 4
            * (
                cos_sigma * (-1 + 2 * cos2_sigma_m**2)
                - bb / 6 * cos2_sigma_m * (-3 + 4 * sin_sigma**2) * (-3 + 4 * cos2_sigma_m**2)
            )
        )
    )
    s = b * aa * (sigma - delta_sigma)

    return s


def h3_stats(geom, h3_res, compact=False):
    cells = geom_to_cell(geom, cell="h3", res=h3_res)
    area = h3.hex_area(h3_res, unit="km^2")
    if compact:
        cells = h3.compact(cells)
    return len(cells), area


def explode_line_to_points(row):
    points = [Point(x) for x in list(row["geometry"].coords)]  # create list of Point objects
    gdf = gpd.GeoDataFrame(
        index=range(len(points)), columns=row.index
    )  # create new GeoDataFrame with all columns and Point geometry
    gdf.loc[:, "geometry"] = points
    gdf.loc[:, row.index.drop("geometry")] = row[row.index.drop("geometry")].values
    return gdf


def ways2geom(ids, url):
    """
    Gets an array of OSM way IDs and returns the
    corresponding array of shapely LineString or Polygon.

    How to call:
    """
    query = "[out:json][timeout:600][maxsize:4073741824];"
    for item in ids:
        query += f"way({item});out geom;"

    response = requests.get(url, params={"data": query}).json()
    response = response["elements"]
    nodes = response[0]["geometry"]  # used later to determine if the way is a Ploygon or a LineString
    ways = [item["geometry"] for item in response]

    geoms = []
    for way in ways:
        coords = [(node["lon"], node["lat"]) for node in way]
        if nodes[0] == nodes[-1]:  # in polygons the first and last items are the same.
            geoms.append(Polygon(coords))
        else:
            geoms.append(LineString(coords))
    return geoms


def way2geom(way_id, url):
    """
    Gets an ID of OSM way object and
    return the corresponding shapely polygon object.
    """
    query = f"[out:json][timeout:600][maxsize:4073741824];way({way_id});out geom;"
    response = requests.get(url, params={"data": query}).json()
    response = response["elements"][0]
    geom = response["geometry"]
    coords = [(node["lon"], node["lat"]) for node in geom]
    if geom[0] == geom[-1]:
        return Polygon(coords)
    else:
        return LineString(coords)


def overlay_parallel(gdf1, gdf2, how="intersection", keep_geom_type=False):
    n_cores = cpu_count()
    gdf1_chunks = np.array_split(gdf1, n_cores)
    gdf2_chunks = [gdf2] * n_cores
    inputs = zip(gdf1_chunks, gdf2_chunks, [how] * n_cores, [keep_geom_type] * n_cores)

    with Pool(n_cores) as pool:  # Create a multiprocessing pool and apply the overlay function in parallel on each chunk
        df = pd.concat(pool.starmap(gpd.overlay, inputs))
    return gpd.GeoDataFrame(df, crs=gdf1.crs)


def decode(encoded):  # decode a route shape in Valhalla
    """
    https://github.com/valhalla/valhalla-docs/blob/master/decoding.md#decode-a-route-shape
    Valhalla routing, map-matching, and elevation services use an encoded polyline format to store a series of latitude,
    longitude coordinates as a single string. Polyline encoding greatly reduces the size of the route response or
    map-matching request, especially for longer routes or GPS traces.
    """

    inv = 1.0 / 1e6  # six degrees of precision in valhalla
    decoded = []
    previous = [0, 0]
    i = 0
    while i < len(encoded):  # for each byte
        ll = [0, 0]  # for each coord (lat, lon)
        for j in [0, 1]:
            shift = 0
            byte = 0x20
            while byte >= 0x20:  # keep decoding bytes until you have this coord
                byte = ord(encoded[i]) - 63
                i += 1
                ll[j] |= (byte & 0x1F) << shift
                shift += 5
            # get the final value adding the previous offset and remember it for the next
            ll[j] = previous[j] + (~(ll[j] >> 1) if ll[j] & 1 else (ll[j] >> 1))
            previous[j] = ll[j]
        # scale by the precision and chop off long coords also flip the positions so
        # its the far more standard lon,lat instead of lat,lon
        decoded.append([float("%.6f" % (ll[1] * inv)), float("%.6f" % (ll[0] * inv))])
    return decoded  # hand back the list of coordinates


def map_matching(df, cost, url, format="osrm"):
    meili_head = '{"shape":'  # Providing needed data for the body of Meili's request

    meili_coordinates = df.to_json(orient="records")  # Getting our Pandas DataFrame into a JSON.

    # Those are parameters that you can change according to the Meili's documentation
    # Make sure to choose a proper "costing" e.g. auto, bicycle, pedestrian
    meili_tail = f', "search_radius":150, "shape_match":"map_snap", "costing":"{cost}", "format":"{format}"}}'

    # Combining all the string into a single request
    meili_request_body = meili_head + meili_coordinates + meili_tail

    # Sending a request
    response = requests.post(url, data=meili_request_body, headers={"Content-type": "application/json"})
    if response.status_code == 200:
        return response.json()  # convert the JSON response to a Python dictionary
    else:
        return None
