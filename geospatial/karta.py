import re
from typing import List, Optional, Union

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


def row_polygons(row: pd.Series, karta: folium.Map, fill_color: str, line_width: int, popup_dict: dict = None) -> int:
    """
    Adds a polygon to a Folium map based on the specified parameters and data in the provided row.

    This function creates a polygon (GeoJson) object for the specified row's geometry and adds it to the Folium map (`karta`).
    It allows customization of fill color, line width, and popups. The function also defines style and highlight properties
    for the polygon.

    Parameters
    ----------
    row : pd.Series
        A row of data containing a 'geometry' attribute that defines the polygon shape.

    karta : folium.Map
        A Folium map object to which the polygon will be added.

    fill_color : str
        Column name to determine the fill color of the polygon. If the column is present in the row, the color is extracted
        using the `color_map` function.

    line_width : int
        The width of the border (outline) of the polygon.

    popup_dict : dict, optional
        A dictionary where keys are labels and values are column names in the row. This dictionary is used to create an
        HTML popup with the specified labels and values for the polygon (default is None).

    Returns
    -------
    int
        Returns 0 upon successfully adding the polygon to the map.

    Examples
    --------
    >>> row = pd.Series({"geometry": Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]), "fill_color": "blue"})
    >>> karta = folium.Map(location=[0.5, 0.5], zoom_start=10)
    >>> row_polygons(row, karta, "fill_color", line_width=2)
    0
    """
    # Determine fill color if specified column is present
    if fill_color in row.index:
        fill_color = color_map(row[fill_color])

    # Style function to apply to the polygon
    def style_function(x):
        return {
            "fillColor": fill_color,
            "color": "#000000",  # Border color
            "fillOpacity": 0.25,
            "weight": line_width,
        }

    # Highlight style function when the polygon is hovered over
    def highlight_function(x):
        return {
            "fillColor": fill_color,
            "color": "#000000",  # Border color
            "fillOpacity": 0.5,
            "weight": line_width,
        }

    # Create a popup if a popup dictionary is provided
    if popup_dict is None:
        popup = None
    else:
        popup = ""
        for item in popup_dict:
            popup += "<b>{}</b>: <b>{}</b><br>".format(item, row[popup_dict[item]])

    # Create a GeoJson object from the row's geometry and add it to the map
    gjson = row.geometry.__geo_interface__
    gjson = folium.GeoJson(data=gjson, style_function=style_function, highlight_function=highlight_function, tooltip=popup)
    gjson.add_to(karta)

    return 0


def polygons(
    karta: folium.Map, mdf: pd.DataFrame, fill_color: str, highlight_color: str, line_width: int, popup_dict: dict = None
) -> int:
    """
    Adds multiple polygons from a DataFrame to a Folium map with specified styles and popups.

    This function iterates over the rows of a DataFrame containing geometries and adds each polygon to the given Folium map.
    The polygon styles can be customized with fill and highlight colors, line width, and optional popup information for
    each polygon.

    Parameters
    ----------
    karta : folium.Map
        The Folium map object to which the polygons will be added.

    mdf : pd.DataFrame
        A DataFrame containing polygon geometries in a 'geometry' column and other optional attributes for styling and popup.

    fill_color : str
        The column name or value used to determine the fill color of the polygons. If the column is present in `mdf`, the color
        is extracted using the `color_map` function. Otherwise, it is used as a direct color value.

    highlight_color : str
        The color used to highlight polygons when they are hovered over.

    line_width : int
        The width of the polygon borders (outlines).

    popup_dict : dict, optional
        A dictionary where keys are labels and values are column names in `mdf`. This dictionary is used to create an HTML popup
        with the specified labels and values for each polygon (default is None).

    Returns
    -------
    int
        Returns 0 upon successfully adding the polygons to the map.

    Examples
    --------
    >>> # Example DataFrame
    >>> data = {'geometry': [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]), Polygon([(1, 1), (2, 1), (2, 2), (1, 2)])],
                'color': ['blue', 'green']}
    >>> mdf = pd.DataFrame(data)
    >>> karta = folium.Map(location=[0.5, 0.5], zoom_start=10)
    >>> polygons(karta, mdf, fill_color="color", highlight_color="yellow", line_width=2)
    0
    """
    # Determine fill color if the specified column is present in the DataFrame
    if fill_color in mdf.columns:
        fill_color = color_map(mdf[fill_color].values[0])

    # Define the default style function for polygons
    def style_function(x):
        return {
            "fillColor": fill_color,
            "color": "#000000",  # Border color
            "fillOpacity": 0.25,
            "weight": line_width,
        }

    # Define the highlight style function for polygons when hovered over
    def highlight_function(x):
        return {
            "fillColor": highlight_color,
            "color": "#000000",  # Border color
            "fillOpacity": 0.5,
            "weight": line_width,
        }

    # Iterate over each row in the DataFrame and add the corresponding polygon to the map
    for _, row in mdf.iterrows():
        # Create a popup if a popup dictionary is provided
        if popup_dict is None:
            popup = None
        else:
            popup = ""
            for item in popup_dict:
                popup += "<b>{}</b>: <b>{}</b><br>".format(item, row[popup_dict[item]])

        # Convert geometry to GeoJson and add it to the Folium map
        gjson = gpd.GeoSeries(row["geometry"]).to_json()
        gjson = folium.GeoJson(data=gjson, style_function=style_function, highlight_function=highlight_function, tooltip=popup)
        gjson.add_to(karta)

    return 0


def plp(  # plp: points, lines, polygons
    gdf_list: Union[pd.DataFrame, gpd.GeoDataFrame, List[Union[pd.DataFrame, gpd.GeoDataFrame]]] = None,
    # Point
    x: Optional[str] = None,
    y: Optional[str] = None,  # provide x and y if more than one column in gdf contains 'lat' and 'lon'
    cluster: bool = False,
    heatmap: bool = False,
    line: bool = False,
    antpath: bool = False,
    point_color: str = "blue",
    color_head: Optional[str] = None,
    color_tail: Optional[str] = None,  # color_head and color_tail: substring indices
    point_opacity: float = 0.5,
    point_radius: int = 3,
    point_weight: int = 6,
    point_popup: Optional[dict] = None,  # point_weight = 2xpoint_radius
    buffer_radius: int = 0,
    ring_inner_radius: int = 0,
    ring_outer_radius: int = 0,
    # LineString
    line_color: str = "blue",
    line_opacity: float = 0.5,
    line_weight: int = 6,
    line_popup: Optional[dict] = None,
    # Polygon
    centroid: bool = False,  # if centroid=True it shows centroids of polygons on the map.
    fill_color: str = "red",
    highlight_color: str = "green",
    line_width: float = 0.3,
    polygon_popup: Optional[dict] = None,
    choropleth_cols: Optional[Union[str, List[str]]] = None,
    choropleth_bins: Optional[List[int]] = None,
    choropleth_legend: Optional[str] = None,
    choropleth_palette: str = "YlOrRd",
    choropleth_highlight: bool = True,
    geohash_res: int = 0,
    s2_res: int = -1,
    h3_res: int = -1,
    geohash_inner: bool = False,
    compact: bool = False,
    cells: Optional[List[str]] = None,
    cell_type: Optional[str] = None,  # list of geohash, S2 or H3 cell IDs
    ways: Optional[List[int]] = None,  # list of OSM way IDs (lines or polygons) and Overpass API URL to query from
    url: Optional[str] = None,  # OpenStreetMap server URL
) -> folium.Map:
    """
    Creates a Folium map with points, lines, or polygons based on the input geospatial data.

    The function `plp` allows users to add different geometrical elements (points, lines, polygons) to a Folium map.
    It supports various visual styles and configurations, such as clustering, heatmaps, choropleth, and geohash or cell-based layers.

    Parameters
    ----------
    gdf_list : list of gpd.GeoDataFrame or pd.DataFrame, optional
        List of GeoDataFrames or DataFrames containing geometrical data to be plotted. If a single DataFrame is provided,
        it will be wrapped in a list internally.

    x : str, optional
        Column name for the x-coordinate (longitude). Required if more than one column in the DataFrame contains 'lon' or 'lat'.

    y : str, optional
        Column name for the y-coordinate (latitude). Required if more than one column in the DataFrame contains 'lon' or 'lat'.

    cluster : bool, default False
        If True, clusters points together based on their proximity using Folium's `MarkerCluster`.

    heatmap : bool, default False
        If True, creates a heatmap layer using Folium's `HeatMap` for points.

    line : bool, default False
        If True, connects points using Folium's `PolyLine` to form lines.

    antpath : bool, default False
        If True, creates animated ant paths for the line geometries using Folium's `AntPath`.

    point_color : str, default "blue"
        Color of the points when displayed on the map.

    color_head : str, optional
        Substring to extract color for the head.

    color_tail : str, optional
        Substring to extract color for the tail.

    point_opacity : float, default 0.5
        Opacity of the points. Value should be between 0 and 1.

    point_radius : int, default 3
        Radius of the points in pixels.

    point_weight : int, default 6
        Weight (thickness) of the point outline. Typically set to twice the `point_radius`.

    point_popup : dict, optional
        Dictionary where keys are labels and values are column names in the DataFrame. Used to create HTML popups with
        attributes of each point.

    buffer_radius : float, default 0
        Buffer radius (in meters) to create a buffer around each point. Set to 0 to disable buffering.

    ring_inner_radius : float, default 0
        Inner radius of ring buffers around points. Only used if `ring_outer_radius` is set.

    ring_outer_radius : float, default 0
        Outer radius of ring buffers around points. If set, creates a ring around each point.

    line_color : str, default "blue"
        Color of the lines connecting points or LineString geometries.

    line_opacity : float, default 0.5
        Opacity of the lines. Value should be between 0 and 1.

    line_weight : int, default 6
        Thickness of the lines.

    line_popup : dict, optional
        Dictionary where keys are labels and values are column names in the DataFrame. Used to create HTML popups with
        attributes of each line.

    centroid : bool, default False
        If True, displays the centroids of polygon geometries on the map.

    fill_color : str, default "red"
        Fill color for polygon geometries.

    highlight_color : str, default "green"
        Color used to highlight polygons when hovered.

    line_width : float, default 0.3
        Thickness of polygon outlines.

    polygon_popup : dict, optional
        Dictionary where keys are labels and values are column names in the DataFrame. Used to create HTML popups with
        attributes of each polygon.

    choropleth_cols : list of str, optional
        List of column names to use for creating choropleth maps. Requires at least two columns for {key, value}.

    choropleth_bins : list of int or float, optional
        Binning strategy for choropleth maps.

    choropleth_legend : str, optional
        Legend title for choropleth maps.

    choropleth_palette : str, default "YlOrRd"
        Color palette for choropleth maps.

    choropleth_highlight : bool, default True
        If True, highlights choropleth polygons when hovered.

    geohash_res : int, default 0
        Resolution for creating geohash-based polygonal layers. Set to 0 to disable.

    s2_res : int, default -1
        Resolution for creating Google S2-based polygonal layers. Set to -1 to disable.

    h3_res : int, default -1
        Resolution for creating Uber H3-based polygonal layers. Set to -1 to disable.

    geohash_inner : bool, default False
        If True, shows only inner geohash cells. Does not work if `compact` is set to True.

    compact : bool, default False
        If True, creates compact representation of geohash, S2, or H3 cells.

    cells : list, optional
        List of geohash, S2, or H3 cell IDs to visualize.

    cell_type : str, optional
        Type of cells used in `cells` parameter. Can be 'geohash', 's2', or 'h3'.

    ways : list of str, optional
        List of OSM way IDs to visualize as lines or polygons.

    url : str, optional
        Overpass API URL to query OSM geometries by `ways` parameter.

    Returns
    -------
    folium.Map
        A Folium map object with the added geometrical features based on input parameters.

    Examples
    --------
    >>> # Example usage
    >>> gdf = gpd.read_file("path/to/shapefile.shp")
    >>> plp(gdf)
    """

    # Handle `cells` input by converting cell IDs to geometries
    if cells:
        res, geoms = sindex.cell_to_geom(cells, cell_type=cell_type)
        gdf = gpd.GeoDataFrame({"id": cells, "res": res, "geometry": geoms}, crs="EPSG:4326")
        karta = plp(gdf, polygon_popup={"ID": "id", "Resolution": "res"})
        return karta

    # Handle `ways` input by converting OSM way IDs to geometries
    if ways:
        geoms = osm.ways_to_geom(ways, url)
        gdf = gpd.GeoDataFrame({"way_id": ways, "geometry": geoms}, crs="EPSG:4326")
        if isinstance(gdf.geometry[0], LineString):
            karta = plp(gdf, line_popup={"way_id": "way_id"}, line_color="way_id")
        else:
            karta = plp(gdf, polygon_popup={"way_id": "way_id"}, fill_color="way_id")
        return karta

    # Ensure `gdf_list` is always a list of GeoDataFrames or DataFrames
    if isinstance(gdf_list, pd.DataFrame):
        gdf_list = [gdf_list]

    # Initialize bounding box coordinates for the map
    minlat, maxlat, minlon, maxlon = 90, -90, 180, -180
    # Iterate through the list of GeoDataFrames to update bounding box
    for gdf in gdf_list:
        if not isinstance(gdf, gpd.GeoDataFrame):
            if not x:  # Determine longitude and latitude columns if x is not specified
                xx = [col for col in gdf.columns if "lon" in col.lower() or "lng" in col.lower()][0]
                yy = [col for col in gdf.columns if "lat" in col.lower()][0]
            lons = gdf[xx]
            lats = gdf[yy]
            minlatg, minlong, maxlatg, maxlong = min(lats), min(lons), max(lats), max(lons)  # minlatg: minlat in gdf
        else:  # If input is a GeoDataFrame, use total_bounds to get the bounding box
            minlong, minlatg, maxlong, maxlatg = gdf.total_bounds
        # Update overall bounding box
        minlat, minlon = min(minlat, minlatg), min(minlon, minlong)
        maxlat, maxlon = max(maxlat, maxlatg), max(maxlon, maxlong)

    # Create a base map using the bounding box
    sw = [minlat, minlon]  # South West (bottom left corner)
    ne = [maxlat, maxlon]  # North East (top right corner)
    karta = base_map(sw, ne)  # Initialize folium map with the bounding box

    # Iterate through each DataFrame or GeoDataFrame in the list to add layers to the map
    for i, gdf in enumerate(gdf_list, start=1):
        geom = gdf.geometry.values[0] if isinstance(gdf, gpd.GeoDataFrame) else None
        # i = 0  # index of gdf in gdf_list
        # for gdf in gdf_list:
        #    i += 1
        #    if not isinstance(gdf, gpd.GeoDataFrame):  # if pd.DataFrame
        #        geom = None
        #    else:
        #        geom = gdf.geometry.values[0]

        # Handle Polygon geometries
        if isinstance(geom, Polygon) or isinstance(geom, MultiPolygon):
            if centroid:  # Show centroids of polygons if `centroid=True`
                group_centroid = folium.FeatureGroup(name=f"{i}- Centroid")
                cdf = gpd.GeoDataFrame({"geometry": gdf.centroid}, crs="EPSG:4326")  # centroid df
                cdf.apply(points, karta=group_centroid, color="red", axis=1)
                group_centroid.add_to(karta)
            # If choropleth columns are provided, visualize as a choropleth map
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
            else:  # Otherwise, visualise polygons normally
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
        # Handle LineString geometries
        if isinstance(geom, LineString):
            group_line = folium.FeatureGroup(name=f"{i}- Line")
            for _index, row in gdf.iterrows():
                coordinates = [
                    (coord[1], coord[0]) for coord in row.geometry.coords
                ]  # Convert LineString geometries to coordinates (lat, lon)
                # Use color mapping if line_color is a column
                color = color_map(row[line_color]) if line_color in gdf.columns else line_color

                # Create popup content if specified
                popup = "".join(f"{item}: <b>{row[line_popup[item]]}</b><br>" for item in line_popup) if line_popup else None
                # if line_popup is None:
                #    popup = None
                # else:
                #    popup = ""
                #    for item in line_popup:
                #        popup += "{}: <b>{}</b><br>".format(item, row[line_popup[item]])
                group_line.add_child(
                    folium.PolyLine(coordinates, color=color, weight=line_weight, opacity=line_opacity, tooltip=popup)
                )
            group_line.add_to(karta)

        # Handle Point geometries or DataFrame inputs with coordinates
        if not isinstance(gdf, gpd.GeoDataFrame) or isinstance(geom, Point):
            if not isinstance(gdf, gpd.GeoDataFrame) and not x:  # If DataFrame and no x specified
                xx = [col for col in gdf.columns if "lon" in col.lower() or "lng" in col.lower()][0]
                yy = [col for col in gdf.columns if "lat" in col.lower()][0]
            else:
                xx = yy = None

            # Create point layers and visualizations
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

            # Add clustering, heatmap, line connections, and buffer/ring visualizations as specified

            # Create a clustering layer if `cluster=True`
            if cluster:
                group_cluster = folium.FeatureGroup(name=f"{i}- Cluster")
                # If the input is a regular DataFrame, use the latitude and longitude columns
                if not isinstance(gdf, gpd.GeoDataFrame):
                    group_cluster.add_child(plugins.MarkerCluster(locations=list(zip(lats, lons))))
                # If it's a GeoDataFrame, use geometry coordinates
                else:
                    group_cluster.add_child(plugins.MarkerCluster(locations=list(zip(gdf.geometry.y, gdf.geometry.x))))
                # Add the clustering layer to the map
                group_cluster.add_to(karta)

            # Create a heatmap layer if `heatmap=True`
            if heatmap:
                group_heatmap = folium.FeatureGroup(name=f"{i}- Heatmap")
                if not isinstance(gdf, gpd.GeoDataFrame):
                    group_heatmap.add_child(plugins.HeatMap(list(zip(lats, lons)), radius=10))
                else:
                    group_heatmap.add_child(plugins.HeatMap(list(zip(gdf.geometry.y, gdf.geometry.x)), radius=10))
                group_heatmap.add_to(karta)

            # Create a line connection layer if `line=True`
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
                # Add the line layer to the map
                group_line.add_to(karta)

            # Create an animated path layer using AntPath if `antpath=True`
            if antpath:
                group_antpath = folium.FeatureGroup(name=f"{i}- Ant Path")
                if not isinstance(gdf, gpd.GeoDataFrame):
                    group_antpath.add_child(plugins.AntPath(list(zip(lats, lons))))
                else:
                    group_antpath.add_child(plugins.AntPath(list(zip(gdf.geometry.y, gdf.geometry.x))))
                group_antpath.add_to(karta)

            # Create a buffer visualization if `buffer_radius > 0`
            if buffer_radius > 0:
                group_buffer = folium.FeatureGroup(name=f"{i}- Buffer")
                bgdf = gdf.copy()  # buffered gdf: Create a copy of the GeoDataFrame to modify geometries
                # Apply buffer to geometries using the specified radius in meters
                bgdf["geometry"] = bgdf.to_crs(gsp.find_proj(gdf.geometry.values[0])).buffer(buffer_radius).to_crs("EPSG:4326")
                # Add the buffered geometries to the map as polygons
                polygons(
                    karta=group_buffer,
                    mdf=bgdf,
                    fill_color=fill_color,
                    highlight_color=fill_color,
                    line_width=line_width,
                    popup_dict=None,
                )
                # Add the buffer layer to the map
                group_buffer.add_to(karta)

            # Create ring visualization if `ring_outer_radius > 0`
            if ring_outer_radius > 0:
                group_ring = folium.FeatureGroup(name=f"{i}- Ring")
                bgdf = gdf.copy()  # buffered gdf: Create a copy of the GeoDataFrame to modify geometries
                # Create ring shapes by applying an outer and inner buffer, subtracting the inner from the outer
                bgdf["geometry"] = (
                    bgdf.to_crs(gsp.find_proj(gdf.geometry.values[0]))
                    .buffer(ring_outer_radius)
                    .difference(bgdf.to_crs(gsp.find_proj(gdf.geometry.values[0])).buffer(ring_inner_radius))
                    .to_crs("EPSG:4326")
                )  # radius in meters
                # Add the ring-shaped geometries to the map as polygons
                polygons(
                    karta=group_ring,
                    mdf=bgdf,
                    fill_color=fill_color,
                    highlight_color=fill_color,
                    line_width=line_width,
                    popup_dict=None,
                )
                # Add the ring layer to the map
                group_ring.add_to(karta)

    # Geohash visualization if `geohash_res > 0`
    if geohash_res > 0:  # inner=False doesn't work if compact=True
        # Create a polygon for bounding box if input is not a polygon
        if isinstance(geom, Polygon) or isinstance(geom, MultiPolygon):
            cdf = gdf.copy()
        else:
            bb = Polygon([[minlon, minlat], [maxlon, minlat], [maxlon, maxlat], [minlon, maxlat], [minlon, minlat]])
            cdf = gpd.GeoDataFrame({"geometry": [bb]}, crs="EPSG:4326")  # Create a bounding box GeoDataFrame

        # Convert geometries to geohash cells and their geometries
        cells, _ = sindex.geom_to_cell_parallel(cdf, cell_type="geohash", res=geohash_res, compact=compact)
        res, geoms = sindex.cell_to_geom(cells, cell_type="geohash")
        cdf = gpd.GeoDataFrame({"id": cells, "res": res, "geometry": geoms}, crs="EPSG:4326")

        # Add geohash cells to the map as a polygon layer
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

    # S2 cell visualization if `s2_res > -1`
    if s2_res > -1:
        # Create a polygon for bounding box if input is not a polygon
        if isinstance(geom, Polygon) or isinstance(geom, MultiPolygon):
            cdf = gdf.copy()
        else:
            bb = Polygon([[minlon, minlat], [maxlon, minlat], [maxlon, maxlat], [minlon, maxlat], [minlon, minlat]])
            cdf = gpd.GeoDataFrame({"geometry": [bb]}, crs="EPSG:4326")  # cell df

        # Convert geometries to S2 cells and their geometries
        cells, _ = sindex.geom_to_cell_parallel(cdf, cell_type="s2", res=s2_res, compact=compact)
        res, geoms = sindex.cell_to_geom(cells, cell_type="s2")
        cdf = gpd.GeoDataFrame({"id": cells, "res": res, "geometry": geoms}, crs="EPSG:4326")

        # Add S2 cells to the map as a polygon layer
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

    # H3 cell visualization if `h3_res > -1`
    if h3_res > -1:
        if isinstance(geom, Polygon) or isinstance(geom, MultiPolygon):
            cdf = gdf.copy()
        # Create a bounding box GeoDataFrame
        else:
            bb = Polygon([[minlon, minlat], [maxlon, minlat], [maxlon, maxlat], [minlon, maxlat], [minlon, minlat]])
            cdf = gpd.GeoDataFrame({"geometry": [bb]}, crs="EPSG:4326")  # cell df

        # Convert geometries to H3 cells and their geometries
        cells, _ = sindex.geom_to_cell_parallel(cdf, cell_type="h3", res=h3_res, compact=compact)
        res, geoms = sindex.cell_to_geom(cells, cell_type="h3")
        cdf = gpd.GeoDataFrame({"id": cells, "res": res, "geometry": geoms}, crs="EPSG:4326")

        # Add H3 cells to the map as a polygon layer
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


def choropleth_plp(
    karta: folium.Map, gdf: gpd.GeoDataFrame, columns: list, bins: list, legend: str, palette: str, highlight: bool
) -> int:
    """
    Adds a choropleth layer to a Folium map using the specified GeoDataFrame and column properties.

    This function is used exclusively within the `plp` function to create choropleth maps, visualizing data attributes
    over a geographic area using color gradients.

    Parameters
    ----------
    karta : folium.Map
        The Folium map object to which the choropleth layer will be added.
    gdf : geopandas.GeoDataFrame
        The GeoDataFrame containing multipolygon geometries and the data to be visualized.
    columns : list
        A list containing two elements:
            - columns[0] : str
                The column name in `gdf` that contains unique identifiers for each region or geometry.
            - columns[1] : str
                The column name in `gdf` containing the data values to be visualized on the map.
    bins : list
        A list of numerical values defining the value intervals to use for the choropleth color categories.
    legend : str
        A string representing the legend title that describes what is being represented on the map (e.g., "Population Density").
    palette : str
        A string defining the color palette to be used for the choropleth (e.g., "YlOrRd", "BuPu").
    highlight : bool
        A boolean flag indicating whether regions should be highlighted when hovered over.

    Returns
    -------
    int
        Returns 0 upon successful execution, indicating that the choropleth layer was successfully added to the map.

    Examples
    --------
    >>> choropleth_plp(
            karta,
            gdf,
            ['region_id', 'population'],
            bins=[0, 100, 500, 1000, 5000],
            legend="Population by Region",
            palette="YlOrRd",
            highlight=True
        )
    """
    # Create a choropleth layer based on the GeoDataFrame, using the specified columns and styling options
    choropleth = folium.Choropleth(
        geo_data=gdf,  # The GeoDataFrame containing geographic data and properties
        name="Choropleth",  # Name of the layer to display in the layer control
        data=gdf,  # Data source used to extract the values to be represented
        columns=columns,  # [unique_identifier_column, data_value_column] to match regions with data
        key_on="feature.properties." + columns[0],  # Key to match regions in the GeoDataFrame with those in geo_data
        legend_name=legend,  # Description of the data being visualized
        bins=bins,  # Value ranges for choropleth colors
        fill_color=palette,  # Color scheme for the choropleth
        fill_opacity=0.5,  # Transparency level of the filled regions
        line_opacity=0.25,  # Transparency level of the borders between regions
        smooth_factor=0,  # Level of smoothing applied to the edges of regions
        highlight=highlight,  # Enable or disable highlighting of regions on hover
    ).geojson.add_to(karta)

    # Add a tooltip to display the attribute values for each region when hovered over
    folium.features.GeoJsonTooltip(fields=columns).add_to(choropleth)

    # Return 0 to indicate successful addition of the choropleth layer to the map
    return 0


def choropleth(
    gdf: gpd.GeoDataFrame, columns: list, bins: list, legend: str, palette: str = "YlOrRd", highlight: bool = True
) -> folium.Map:
    """
    Creates a choropleth map based on the given GeoDataFrame and specified parameters.

    This function generates a Folium choropleth map layer using the data from a GeoDataFrame and visualizes it using color
    gradients to represent different data values across geographic areas.

    Args:
        gdf (geopandas.GeoDataFrame): The GeoDataFrame containing multipolygon geometries and data attributes to be visualized.
        columns (list): A list of two elements:
            - `columns[0]` (str): The column name in `gdf` that contains unique identifiers for each region.
            - `columns[1]` (str): The column name in `gdf` containing the data values to be visualized.
        bins (list): A list of numerical values defining the value intervals for the choropleth color categories.
        legend (str): A string that provides the title for the legend to describe what is represented on the map.
        palette (str, optional): A string defining the color palette to be used for the choropleth (default is "YlOrRd").
        highlight (bool, optional): A boolean flag indicating whether regions should be highlighted on hover (default is True).

    Returns:
        folium.Map: The Folium map object containing the choropleth layer.

    Example:
        choropleth(
            gdf,
            ['region_id', 'population'],
            bins=[0, 100, 500, 1000, 5000],
            legend="Population by Region",
            palette="YlOrRd",
            highlight=True
        )
    """

    # Extract the bounding coordinates of the GeoDataFrame
    minlon, minlat, maxlon, maxlat = gdf.total_bounds  # Get the total bounds of the GeoDataFrame
    sw = [minlat, minlon]  # South-west corner
    ne = [maxlat, maxlon]  # North-east corner
    karta = base_map(sw, ne)  # Create a base map using the bounding coordinates

    # Create a choropleth layer based on the GeoDataFrame
    choropleth = folium.Choropleth(
        geo_data=gdf,  # The GeoDataFrame containing geographic data
        name="Choropleth",  # Name of the layer for display in layer control
        data=gdf,  # The data source for values to be represented
        columns=columns,  # [unique_identifier_column, data_value_column] for matching regions with data
        key_on="feature.properties." + columns[0],  # Key to match GeoDataFrame regions with the data
        legend_name=legend,  # Description of the data being visualized
        bins=bins,  # Value ranges for choropleth colors
        fill_color=palette,  # Color scheme for the choropleth
        fill_opacity=0.5,  # Transparency level of filled regions
        line_opacity=0.25,  # Transparency level of borders between regions
        smooth_factor=0,  # Level of smoothing applied to the edges of regions
        highlight=highlight,  # Enable or disable highlighting of regions on hover
    ).add_to(karta)  # Add the choropleth layer to the map

    # Add a tooltip to display the attribute values for each region when hovered over
    folium.features.GeoJsonTooltip(fields=columns).add_to(choropleth.geojson)

    # Add layer control to the map
    folium.LayerControl(collapsed=False).add_to(karta)

    # Return the Folium map object containing the choropleth layer
    return karta
