import math
from math import atan2, cos, radians, sin, sqrt
from multiprocessing import Pool, cpu_count
from typing import List, Optional, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
import requests
from shapely.geometry import MultiPolygon, Point, Polygon
from shapely.ops import transform

# pd.options.mode.chained_assignment = None  # default='warn'


def geom_stats(geom: Optional[Union[Polygon, MultiPolygon]] = None, unit: str = "m") -> Optional[List[Union[int, float]]]:
    """
    Computes various geometric statistics for a given Polygon or MultiPolygon geometry, including the number of
    shells (outer boundaries), number of holes, number of shell points, total area, and total border length.

    If no geometry is provided, the function prints a usage example.

    Args:
        geom (Optional[Union[Polygon, MultiPolygon]]): A Shapely geometry object (Polygon or MultiPolygon)
                                                       for which to compute the statistics. Defaults to None.
        unit (str): The unit for area and length calculations. Accepts "m" for meters and "km" for kilometers.
                    Defaults to "m".

    Returns:
        Optional[List[Union[int, float]]]: A list containing the following statistics in order:
                                           - Number of shells (int)
                                           - Number of holes (int)
                                           - Number of shell points (int)
                                           - Total area (float, rounded to nearest integer in specified unit)
                                           - Total border length (float, rounded to nearest integer in specified unit)

        Returns None if no geometry is provided.
    """
    if not geom:  # Print usage help if geom is None
        print(
            "mdf[['nshells', 'nholes', 'nshell_points', 'area', 'border']] = [gsp.geom_stats(geom, unit='km') for geom in mdf.geometry]"
        )
        return

    # Determine the appropriate UTM zone for the given geometry
    utm_zone = find_proj(geom)

    # Handle different geometry types
    if geom.geom_type == "Polygon":
        polylist = [geom]
    elif geom.geom_type == "MultiPolygon":
        polylist = list(geom.geoms)
    else:
        raise ValueError("The input geometry must be a Polygon or MultiPolygon.")

    # Initialize variables for calculating statistics
    n_shells = len(polylist)
    n_holes = n_shell_points = border = area = 0

    # Iterate through each Polygon in the list to calculate statistics
    for poly in polylist:
        n_holes += len(poly.interiors)  # Count the number of holes
        n_shell_points += len(poly.exterior.coords)  # Count the number of shell points
        # Transform geometry to the appropriate UTM zone and calculate length/area
        border += trans_proj(poly, "EPSG:4326", utm_zone).exterior.length
        area += trans_proj(poly, "EPSG:4326", utm_zone).area

    # Return statistics based on the specified unit
    if unit == "m":  # If unit is meters
        return [n_shells, n_holes, n_shell_points, round(area), round(border)]
    else:  # If unit is kilometers
        return [n_shells, n_holes, n_shell_points, round(area / 1_000_000), round(border / 1000)]


def find_proj(geom: Union[Point, Polygon, MultiPolygon]) -> str:
    """
    Determines the appropriate Universal Transverse Mercator (UTM) zone projection
    based on the input geometry's centroid coordinates.

    The UTM is a map projection system that divides the Earth into multiple zones,
    each with a specific coordinate reference system (CRS). The function returns
    the EPSG code corresponding to the UTM zone of the geometry.

    Args:
        geom (Union[Point, Polygon, MultiPolygon]): A Shapely geometry object, which
                                                    can be a Point, Polygon, or MultiPolygon.

    Returns:
        str: The EPSG code representing the UTM projection for the geometry's location.
             For northern hemisphere, it returns codes in the format 'EPSG:326XX', and for
             southern hemisphere, it returns 'EPSG:327XX', where 'XX' is the UTM zone number.
    """
    if geom.geom_type != "Point":
        # If the geometry is not a Point, use its centroid
        geom = geom.centroid

    # Extract latitude and longitude from the geometry
    lat = geom.y
    lon = geom.x

    # Determine the base EPSG code depending on the hemisphere
    if lat >= 0:
        proj = "EPSG:326"  # Northern Hemisphere
    else:
        proj = "EPSG:327"  # Southern Hemisphere

    # Calculate the UTM zone number based on longitude
    utm = math.ceil(30 + lon / 6)

    # Return the complete EPSG code for the UTM projection
    return proj + str(utm)


def trans_proj(geom: base.BaseGeometry, proj1: str, proj2: str) -> base.BaseGeometry:
    """
    Transforms a geometry object from one coordinate reference system (CRS) to another.

    This function uses the `pyproj` library to create a transformation pipeline that converts
    the input geometry from the source CRS (`proj1`) to the target CRS (`proj2`). The resulting
    geometry is returned in the new projection.

    Args:
        geom (base.BaseGeometry): The Shapely geometry object to be transformed. This can be
                                  any geometry type (e.g., Point, Polygon).
        proj1 (str): The EPSG code or PROJ string representing the source CRS of the input geometry.
        proj2 (str): The EPSG code or PROJ string representing the target CRS for the transformed geometry.

    Returns:
        base.BaseGeometry: The transformed Shapely geometry object in the new projection.
    """
    # Create a transformation function using pyproj's Transformer
    project = pyproj.Transformer.from_crs(pyproj.CRS(proj1), pyproj.CRS(proj2), always_xy=True).transform

    # Apply the transformation to the geometry and return the transformed geometry
    return transform(project, geom)


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
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


def vincenty(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
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


def flatten_3d(geom: gpd.GeoSeries) -> List[Union[Polygon, MultiPolygon]]:
    """
    Converts a GeoSeries of 3D Polygons (Polygon Z) or MultiPolygons into a list of 2D Polygons or MultiPolygons
    by removing the z-coordinate from each geometry.

    Args:
        geom (gpd.GeoSeries): A GeoSeries containing 3D Polygons or MultiPolygons (geometries with z-coordinates).

    Returns:
        List[Union[Polygon, MultiPolygon]]: A list of 2D Polygons or MultiPolygons with z-coordinates removed.

    Example:
        >>> gdf.geometry = gsp.flatten_3d(gdf.geometry)
    """
    new_geom = []
    for p in geom:
        if p.has_z:
            if p.geom_type == "Polygon":
                lines = [xy[:2] for xy in list(p.exterior.coords)]
                new_p = Polygon(lines)
                new_geom.append(new_p)
            elif p.geom_type == "MultiPolygon":
                new_multi_p = []
                for ap in p:
                    lines = [xy[:2] for xy in list(ap.exterior.coords)]
                    new_p = Polygon(lines)
                    new_multi_p.append(new_p)
                new_geom.append(MultiPolygon(new_multi_p))
    return new_geom


def explode_line_to_points(row: gpd.GeoSeries) -> gpd.GeoDataFrame:
    """
    Splits a LineString geometry from a GeoSeries row into individual Point geometries and returns a new
    GeoDataFrame with each Point as a separate row while preserving the original attributes.

    Args:
        row (gpd.GeoSeries): A GeoSeries representing a single row of a GeoDataFrame.
                             It must include a 'geometry' column of type LineString.

    Returns:
        gpd.GeoDataFrame: A new GeoDataFrame where each row corresponds to a Point geometry derived
                          from the coordinates of the LineString. All other columns from the original row are preserved.
    """
    points = [Point(x) for x in list(row["geometry"].coords)]  # create list of Point objects
    gdf = gpd.GeoDataFrame(
        index=range(len(points)), columns=row.index
    )  # create new GeoDataFrame with all columns and Point geometry
    gdf.loc[:, "geometry"] = points
    gdf.loc[:, row.index.drop("geometry")] = row[row.index.drop("geometry")].values
    return gdf


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


def overlay_parallel(
    gdf1: gpd.GeoDataFrame, gdf2: gpd.GeoDataFrame, how: str = "intersection", keep_geom_type: bool = False
) -> gpd.GeoDataFrame:
    """
    Perform a spatial overlay operation between two GeoDataFrames in parallel using multiple CPU cores.

    The function splits the first GeoDataFrame into chunks based on the number of available CPU cores and
    applies the overlay operation (e.g., intersection, union, difference) in parallel on each chunk with
    respect to the second GeoDataFrame. The results are then concatenated and returned as a single GeoDataFrame.

    Args:
        gdf1 (gpd.GeoDataFrame): The first GeoDataFrame to be used in the spatial overlay operation.
        gdf2 (gpd.GeoDataFrame): The second GeoDataFrame to be used in the spatial overlay operation.
        how (str, optional): The type of overlay operation to perform. Options include "intersection", "union",
                             "difference", "symmetric_difference", and "identity". Defaults to "intersection".
        keep_geom_type (bool, optional): Whether to retain the original geometry type (e.g., Polygon, LineString)
                                         in the resulting overlay. If set to True, only features of the same
                                         geometry type are retained. Defaults to False.

    Returns:
        gpd.GeoDataFrame: A new GeoDataFrame resulting from the spatial overlay operation, with the same coordinate
                          reference system (CRS) as the first input GeoDataFrame (`gdf1`).
    """
    # Determine the number of CPU cores available for parallel processing
    n_cores = cpu_count()

    # Split the first GeoDataFrame into chunks for parallel processing
    gdf1_chunks = np.array_split(gdf1, n_cores)

    # Create a list of the second GeoDataFrame repeated for each chunk
    gdf2_chunks = [gdf2] * n_cores

    # Prepare inputs for the parallel processing pool
    inputs = zip(gdf1_chunks, gdf2_chunks, [how] * n_cores, [keep_geom_type] * n_cores)

    # Create a multiprocessing pool and apply the overlay function in parallel on each chunk
    with Pool(n_cores) as pool:
        df = pd.concat(pool.starmap(gpd.overlay, inputs))

    # Return the concatenated GeoDataFrame with the same CRS as the first input GeoDataFrame
    return gpd.GeoDataFrame(df, crs=gdf1.crs)


def google_geocoding(address_or_zipcode: str, api_key: str) -> pd.Series:
    """
    Returns geographic coordinates (latitude and longitude) for a given address or zip code
    using the Google Geocoding API.

    Args:
        address_or_zipcode (str): A text-based address or a zip code that you want to geocode.
        api_key (str): A valid Google Maps API key for accessing the geocoding service.

    Returns:
        pd.Series: A pandas Series containing the latitude and longitude as floats.
                   If the request fails or the address is not found, returns a Series of (None, None).

    Example:
        >>> df[["lat", "lon"]] = df.apply(lambda row: gsp.google_geocoding(row.address), axis=1)
        >>> google_geocoding("1600 Amphitheatre Parkway, Mountain View, CA", "your_api_key")
        lat    37.4224764
        lon   -122.0842499
        dtype: float64
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
        pass  # Handle any errors that may occur
    return pd.Series([lat, lon])


def google_reverse_geocoding(lat: float, lon: float, api_key: str) -> str:
    """
    This function takes a coordinate (lat, lon) and returns the postcode.

    df['postcode'] = df.apply(lambda row : gsp.google_reverse_geocoding(row.lat, row.lon), axis=1)
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
