import re

import folium  # Folium is a Python library used for visualising geospatial data. Actually, it's a Python wrapper for Leaflet which is a leading open-source JavaScript library for plotting interactive maps.
import geopandas as gpd
import pandas as pd
from folium import plugins
from shapely.geometry import LineString, MultiPolygon, Point, Polygon

from geospatial import geospatial as gsp
from geospatial import osm, sindex

# pd.options.mode.chained_assignment = None  # default='warn'


def color_map(col: int or str, head: int = None, tail: int = None) -> str:
    """
    Generates a consistent color based on the input column value by mapping it to a predefined color palette.

    This function uses a set color palette and maps the given column value to a color. If the column value is a string,
    a substring can be selected using `head` and `tail` indices, and it will be converted to a numerical index. If the
    column value is an integer, it will directly be mapped to a color using modulo arithmetic.

    Parameters
    ----------
    col : int or str
        The column value to be mapped to a color. It can be either an integer or a string.
        - If an integer, it is directly used for color mapping.
        - If a string, it will be cleaned of non-alphanumeric characters, and a substring defined by `head` and `tail`
          can be selected for mapping.

    head : int, optional
        The starting index of the substring to be used for color mapping if `col` is a string. Default is None.

    tail : int, optional
        The ending index of the substring to be used for color mapping if `col` is a string. Default is None.

    Returns
    -------
    str
        A hexadecimal color code selected from the predefined palette corresponding to the input column value.


    Examples
    --------
    >>> color_map("Category1")
    '#e6194b'  # Red color from the palette

    >>> color_map(5)
    '#3cb44b'  # Green color from the palette

    >>> color_map("Example", head=0, tail=3)
    '#e12348'  # Bright Red from the palette
    """
    # Predefined color palette
    palettet = [
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

    # Check if the column is an integer or string
    if isinstance(col, int):
        idx = col % len(palettet)  # Get color index using modulo arithmetic
    else:
        col = str(col)  # Convert to string
        col = re.sub(r"[\W_]+", "", col)  # Remove non-alphanumeric characters
        idx = int(col[head:tail], 36) % len(palettet)  # Convert substring to a number base 36 (36 = 10 digits + 26 letters)

    return palettet[idx]


def base_map(sw: list or tuple, ne: list or tuple) -> folium.Map:
    """
    Creates a base map with multiple tile layers and fits the map to the specified bounding box.

    This function initializes a Folium map object with multiple tile layers, including:
    - `Bright Mode` (CartoDB Positron)
    - `Dark Mode` (CartoDB Dark Matter)
    - `Satellite` (Esri World Imagery)
    - `OpenStreetMap` (OSM)

    It then fits the map's view to the bounding box defined by the southwest (`sw`) and northeast (`ne`) coordinates.

    Parameters
    ----------
    sw : list or tuple
        The southwest coordinate [latitude, longitude] of the bounding box to fit the map view.

    ne : list or tuple
        The northeast coordinate [latitude, longitude] of the bounding box to fit the map view.

    Returns
    -------
    folium.Map
        A Folium map object with multiple tile layers and the view fitted to the provided bounding box.

    Examples
    --------
    >>> sw = [51.2652, -0.5426]  # Southwest coordinate (London, UK)
    >>> ne = [51.7225, 0.2824]  # Northeast coordinate (London, UK)
    >>> karta = base_map(sw, ne)
    >>> karta.save("map.html")  # Save the map to an HTML file
    """
    # Initialize the base map without any default tiles
    karta = folium.Map(tiles=None)

    # Dictionary of tile layers to be added
    tiles = {
        "cartodbpositron": "Bright Mode",
        "cartodbdark_matter": "Dark Mode",
    }

    # Add each tile layer to the map
    for item in tiles:
        folium.TileLayer(item, name=tiles[item], max_zoom=21).add_to(karta)

    # Add a satellite tile layer (Esri World Imagery)
    folium.TileLayer(
        name="Satellite",
        attr="Esri",
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        overlay=False,
        control=True,
        max_zoom=19,
    ).add_to(karta)

    # Add OpenStreetMap (OSM) tile layer
    folium.TileLayer("openstreetmap", name="OpenStreetMap", max_zoom=19).add_to(karta)

    # Fit the map's view to the bounding box defined by the southwest and northeast coordinates
    karta.fit_bounds([sw, ne])

    return karta


def points(
    row: pd.Series,
    karta: folium.Map,
    color: str,
    color_head: int = None,
    color_tail: int = None,
    opacity: float = 0.5,
    radius: int = 3,
    weight: int = 6,
    popup_dict: dict = None,
    x: str = None,
    y: str = None,
) -> int:
    """
    Adds a point (marker) to a Folium map based on the specified parameters and data in the provided row.

    The function attempts to extract coordinates from a geometry column if available, or directly from `x` and `y` columns
    (longitude and latitude). It then adds a circle marker to the Folium map (`karta`) using the specified color, radius,
    and other style parameters.

    Parameters
    ----------
    row : pd.Series
        A row of data containing either a 'geometry' attribute or x/y columns for coordinates.

    karta : folium.Map
        A Folium map object to which the marker will be added.

    color : str
        Column name to determine the color of the marker. If the column is present in the row, a color from the color map
        will be used.

    color_head : int, optional
        Starting index for substring extraction from the `color` column value to create a unique color (default is None).

    color_tail : int, optional
        Ending index for substring extraction from the `color` column value to create a unique color (default is None).

    opacity : float, optional
        Opacity of the marker (default is 0.5).

    radius : int, optional
        Radius of the circle marker (default is 3).

    weight : int, optional
        Weight (thickness) of the circle marker's border (default is 6).

    popup_dict : dict, optional
        A dictionary where keys are labels and values are column names in the row. This dictionary is used to create
        an HTML popup with the specified labels and values (default is None).

    x : str, optional
        Column name for longitude, if 'geometry' attribute is not present (default is None).

    y : str, optional
        Column name for latitude, if 'geometry' attribute is not present (default is None).

    Returns
    -------
    int
        Returns 0 upon successfully adding the marker to the map.

    Examples
    --------
    >>> row = pd.Series({"geometry": Point(40.748817, -73.985428), "color": "red"})
    >>> karta = folium.Map(location=[40.748817, -73.985428], zoom_start=12)
    >>> points(row, karta, "color")
    0
    """
    try:
        # Attempt to extract coordinates from the geometry column if present
        location = [row.geometry.y, row.geometry.x]
    except Exception:
        # If geometry is not present, use x and y columns for location
        location = [row[y], row[x]]  # x, y: lon, lat column names in DataFrame

    # Determine color if column is specified
    if color in row.index:  # color in DataFrame columns
        color = color_map(row[color], color_head, color_tail)

    # Create a popup HTML if popup_dict is provided
    if popup_dict is None:
        popup = None
    else:
        popup = ""
        for item in popup_dict:
            popup += "{}: <b>{}</b><br>".format(item, row[popup_dict[item]])

    # Add a CircleMarker to the map with the specified parameters
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
        res, geoms = sindex.cell_to_geom(cells, cell_type=cell_type)
        gdf = gpd.GeoDataFrame({"id": cells, "res": res, "geometry": geoms}, crs="EPSG:4326")
        karta = plp(gdf, polygon_popup={"ID": "id", "Resolution": "res"})
        return karta

    if ways:
        geoms = osm.ways_to_geom(ways, url)
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
                    bgdf.to_crs(gsp.find_proj(gdf.geometry.values[0])).buffer(buffer_radius).to_crs("EPSG:4326")
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
                    bgdf.to_crs(gsp.find_proj(gdf.geometry.values[0]))
                    .buffer(ring_outer_radius)
                    .difference(bgdf.to_crs(gsp.find_proj(gdf.geometry.values[0])).buffer(ring_inner_radius))
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

        cells, _ = sindex.geom_to_cell_parallel(cdf, cell_type="geohash", res=geohash_res, compact=compact)
        # ids, res, geoms = sindex.cell_to_geom(cells, cell_type='geohash')
        res, geoms = sindex.cell_to_geom(cells, cell_type="geohash")
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

        cells, _ = sindex.geom_to_cell_parallel(cdf, cell_type="s2", res=s2_res, compact=compact)
        res, geoms = sindex.cell_to_geom(cells, cell_type="s2")
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

        cells, _ = sindex.geom_to_cell_parallel(cdf, cell_type="h3", res=h3_res, compact=compact)
        res, geoms = sindex.cell_to_geom(cells, cell_type="h3")
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
