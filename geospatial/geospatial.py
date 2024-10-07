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
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform

# pd.options.mode.chained_assignment = None  # default='warn'


def geom_stats(geom: Optional[Union[Polygon, MultiPolygon]] = None, unit: str = "m") -> Optional[List[Union[int, float]]]:
    """
    Computes geometric statistics for a Polygon or MultiPolygon geometry.

    Calculates various statistics for a given Shapely geometry, such as the number of shells (outer boundaries),
    number of holes, number of shell points, total area, and total border length. If no geometry is provided,
    the function will print a usage example.

    Parameters
    ----------
    geom : Polygon or MultiPolygon, optional
        A Shapely geometry object (Polygon or MultiPolygon) for which to compute the statistics. If not provided,
        the function will print a usage example and not perform any computations. Default is None.
    unit : str, optional
        The unit for area and length calculations. Accepts "m" for meters and "km" for kilometers. Default is "m".

    Returns
    -------
    list of int or float, optional
        A list containing the following statistics in order:
            - Number of shells (int)
            - Number of holes (int)
            - Number of shell points (int)
            - Total area (float, rounded to nearest integer in the specified unit)
            - Total border length (float, rounded to nearest integer in the specified unit)

        If no geometry is provided, the function returns None.

    Examples
    --------
    >>> from shapely.geometry import Polygon
    >>> geom = Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])
    >>> compute_geometry_statistics(geom, unit="km")
    [1, 0, 5, 1.0, 4.0]
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
    Determines the appropriate UTM zone projection for a given geometry.

    Calculates the Universal Transverse Mercator (UTM) zone projection based on the centroid
    coordinates of the input geometry. The function returns the corresponding EPSG code for
    the UTM zone in which the geometry is located.

    Parameters
    ----------
    geom : Point, Polygon, or MultiPolygon
        A Shapely geometry object, which can be a Point, Polygon, or MultiPolygon.

    Returns
    -------
    str
        The EPSG code representing the UTM projection for the geometry's location. For the
        northern hemisphere, the function returns codes in the format 'EPSG:326XX'. For the
        southern hemisphere, it returns 'EPSG:327XX', where 'XX' is the UTM zone number.

    Notes
    -----
    The UTM (Universal Transverse Mercator) system divides the Earth into 60 longitudinal zones,
    each 6 degrees wide. This function uses the centroid of the input geometry to determine the
    appropriate zone and EPSG code.

    Examples
    --------
    >>> from shapely.geometry import Polygon
    >>> geom = Polygon([(-120, 35), (-121, 35), (-121, 36), (-120, 36), (-120, 35)])
    >>> find_proj(geom)
    'EPSG:32610'
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


def trans_proj(geom: BaseGeometry, proj1: str, proj2: str) -> BaseGeometry:
    """
    Transforms a Shapely geometry object from one CRS to another.

    Uses `pyproj` to create a transformation pipeline that converts the input geometry
    from the source CRS (`proj1`) to the target CRS (`proj2`). The resulting geometry
    is returned in the new coordinate reference system.

    Parameters
    ----------
    geom : BaseGeometry
        A Shapely geometry object to be transformed. This can include Point, Polygon,
        MultiPolygon, LineString, or any other Shapely geometry type.
    proj1 : str
        The EPSG code or PROJ string representing the source CRS of the input geometry.
    proj2 : str
        The EPSG code or PROJ string representing the target CRS for the transformed geometry.

    Returns
    -------
    BaseGeometry
        The transformed Shapely geometry object in the target projection.

    Notes
    -----
    - The function requires `pyproj` and `shapely` libraries.
    - Ensure that the input and output CRS definitions are valid and supported by `pyproj`.

    Examples
    --------
    >>> from shapely.geometry import Point
    >>> geom = Point(10, 50)
    >>> trans_proj(geom, "EPSG:4326", "EPSG:32632")
    <Point object at 0x...>

    """
    # Create a transformation function using pyproj's Transformer
    project = pyproj.Transformer.from_crs(pyproj.CRS(proj1), pyproj.CRS(proj2), always_xy=True).transform

    # Apply the transformation to the geometry and return the transformed geometry
    return transform(project, geom)


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculates the great-circle distance between two points on the Earth's surface.

    The haversine formula determines the shortest distance over the Earth's surface
    between two points given their latitudes and longitudes. The result is the
    distance in meters, based on a mean Earth radius.

    Parameters
    ----------
    lat1 : float
        Latitude of the first point in decimal degrees.
    lon1 : float
        Longitude of the first point in decimal degrees.
    lat2 : float
        Latitude of the second point in decimal degrees.
    lon2 : float
        Longitude of the second point in decimal degrees.

    Returns
    -------
    float
        The great-circle distance between the two points in meters.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Haversine_formula
    .. [2] https://en.wikipedia.org/wiki/Longitude

    Examples
    --------
    >>> haversine(52.2296756, 21.0122287, 41.8919300, 12.5113300)
    1319743.483

    Notes
    -----
    The mean Earth radius is taken as 6,371,008.8 meters.
    a = 6378137.0        # Equatorial radius
    b = 6356752.3142     # Polar radius
    R = (2*a + b)/3      # Mean radius = 6371008.7714
    """
    r = 6371008.8  # Mean Earth radius in meters
    lat1, lat2, dlon = radians(lat1), radians(lat2), radians(lon2 - lon1)
    dlat = lat2 - lat1

    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))  # Angular distance in radians
    return r * c  # Distance in meters


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
    Flattens a GeoSeries of 3D Polygons or MultiPolygons into 2D geometries.

    This function removes the z-coordinate from each 3D geometry in the input GeoSeries,
    converting it into a 2D Polygon or MultiPolygon. The result is a list of 2D geometries.

    Parameters
    ----------
    geom : gpd.GeoSeries
        A GeoSeries containing 3D Polygons or MultiPolygons (geometries with z-coordinates).

    Returns
    -------
    List[Union[Polygon, MultiPolygon]]
        A list of 2D Polygons or MultiPolygons with the z-coordinates removed.

    Examples
    --------
    >>> gdf.geometry = gsp.flatten_3d(gdf.geometry)
        Converts all 3D geometries in the GeoSeries `gdf.geometry` to 2D geometries.

    Notes
    -----
    The function is useful when working with datasets that contain 3D geometries but
    only 2D geometries are needed for further spatial analysis or visualization.

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
    Splits a LineString geometry into individual Point geometries while preserving original attributes.

    This function takes a GeoSeries representing a single row of a GeoDataFrame, extracts the coordinates
    from a LineString geometry, and creates a new GeoDataFrame with each Point as a separate row. All original
    attributes from the input row are preserved in the new GeoDataFrame.

    Parameters
    ----------
    row : gpd.GeoSeries
        A GeoSeries representing a single row of a GeoDataFrame. It must include a 'geometry' column
        containing a LineString geometry.

    Returns
    -------
    gpd.GeoDataFrame
        A new GeoDataFrame where each row corresponds to a Point geometry derived from the coordinates of the LineString.
        All other columns from the original row are preserved.

    Examples
    --------
    >>> line_gdf = gpd.GeoDataFrame({"geometry": [LineString([(0, 0), (1, 1), (2, 2)])]})
    >>> point_gdf = split_linestring_to_points(line_gdf.iloc[0])
    >>> print(point_gdf)
       geometry
    0  POINT (0 0)
    1  POINT (1 1)
    2  POINT (2 2)
    """
    points = [Point(x) for x in list(row["geometry"].coords)]  # create list of Point objects
    gdf = gpd.GeoDataFrame(
        index=range(len(points)), columns=row.index
    )  # create new GeoDataFrame with all columns and Point geometry
    gdf.loc[:, "geometry"] = points
    gdf.loc[:, row.index.drop("geometry")] = row[row.index.drop("geometry")].values
    return gdf


def intersection(gdf1: gpd.GeoDataFrame, gdf2: gpd.GeoDataFrame, poly_id: Optional[str] = None) -> gpd.GeoDataFrame:
    """
    Performs a spatial intersection between two GeoDataFrames and return the intersecting subset of the first GeoDataFrame.

    This function identifies geometries in `gdf1` that intersect with any geometries in `gdf2`. It adds a new column, `counts`,
    to `gdf2` representing the number of intersecting geometries for each feature in `gdf2`. If a `poly_id` column is specified,
    it also adds the geometry ID from `gdf2` to the intersected subset of `gdf1`.

    Parameters
    ----------
    gdf1 : geopandas.GeoDataFrame
        The first GeoDataFrame whose geometries are tested for intersection with `gdf2`.
    gdf2 : geopandas.GeoDataFrame
        The second GeoDataFrame containing geometries to intersect with `gdf1`.
    poly_id : str, optional
        The column name in `gdf2` containing unique geometry identifiers. If provided, the intersected subset of `gdf1`
        will include a new column `geom_id` indicating the geometry ID from `gdf2` that each feature intersects with.

    Returns
    -------
    geopandas.GeoDataFrame
        A new GeoDataFrame containing only the intersecting geometries from `gdf1` with respect to `gdf2`.
        If `poly_id` is provided, the intersected GeoDataFrame will also include a `geom_id` column.

    Examples
    --------
    >>> gdf1 = geopandas.read_file("data1.shp")
    >>> gdf2 = geopandas.read_file("data2.shp")
    >>> result_gdf = intersection(gdf1, gdf2, poly_id="region_id")

    Notes
    -----
    The function modifies `gdf2` in place by adding a `counts` column, which reflects the number of geometries
    in `gdf1` that intersect with each geometry in `gdf2`.

    """
    int_gdf = pd.DataFrame()  # Initialize an empty DataFrame to store intersecting geometries from gdf1
    counts = []  # List to store counts of intersecting geometries for each feature in gdf2

    for geom in gdf2.geometry:
        # Filter `gdf1` to retain only geometries that intersect with the current geometry in `gdf2`
        gdf = gdf1[gdf1.intersects(geom)]

        if poly_id is not None and len(gdf) > 0:
            # If `poly_id` is provided, retrieve the geometry ID from `gdf2` and assign it to `geom_id` column in `gdf`
            gid = gdf2[gdf2.geometry == geom][poly_id].iloc[0]
            gdf["geom_id"] = gid

        # Concatenate the intersecting geometries to the final DataFrame
        int_gdf = pd.concat([int_gdf, gdf])
        counts.append(len(gdf))  # Store the number of intersecting geometries for the current feature in gdf2

    gdf2["counts"] = counts  # Add the counts of intersecting geometries as a new column in gdf2
    return int_gdf  # Return the intersected subset of gdf1


def quick_intersection(gdf1: gpd.GeoDataFrame, gdf2: gpd.GeoDataFrame, poly_id: Optional[str] = None) -> gpd.GeoDataFrame:
    """
    Performs a quick spatial intersection between two GeoDataFrames using bounding box optimization.

    This function identifies geometries in `gdf1` that intersect with any geometries in `gdf2`. It uses
    a spatial index to quickly filter `gdf1` geometries that are likely to intersect with the bounding
    box of each geometry in `gdf2`. It then performs a precise intersection check on this subset, improving
    the performance of the intersection operation.

    If a `poly_id` column is provided, the function adds a new `geom_id` column to the resulting intersected
    GeoDataFrame, storing the geometry ID from `gdf2` that each feature in `gdf1` intersects with. It also
    modifies `gdf2` by adding a `counts` column to indicate the number of intersecting geometries.

    Parameters
    ----------
    gdf1 : geopandas.GeoDataFrame
        The first GeoDataFrame whose geometries are tested for intersection with `gdf2`.
    gdf2 : geopandas.GeoDataFrame
        The second GeoDataFrame containing geometries to intersect with `gdf1`.
    poly_id : str, optional
        The column name in `gdf2` containing unique geometry identifiers. If provided, the intersected subset of `gdf1`
        will include a new column `geom_id` indicating the geometry ID from `gdf2` that each feature intersects with.

    Returns
    -------
    geopandas.GeoDataFrame
        A new GeoDataFrame containing only the intersecting geometries from `gdf1` with respect to `gdf2`.
        If `poly_id` is provided, the intersected GeoDataFrame will also include a `geom_id` column.

    Examples
    --------
    >>> gdf1 = geopandas.read_file("data1.shp")
    >>> gdf2 = geopandas.read_file("data2.shp")
    >>> result_gdf = quick_intersection(gdf1, gdf2, poly_id="region_id")

    Notes
    -----
    - This function modifies `gdf2` in place by adding a `counts` column, which reflects the number of geometries
      in `gdf1` that intersect with each geometry in `gdf2`.
    - It leverages spatial indexing using the `sindex` attribute of `gdf1` to quickly identify candidates for
      intersection, which significantly improves performance for large datasets.

    """
    int_gdf = pd.DataFrame()  # Initialize an empty DataFrame to store intersecting geometries from gdf1
    counts = []  # List to store counts of intersecting geometries for each feature in gdf2

    for geom in gdf2.geometry:
        # Get the indices of geometries in `gdf1` that are likely to intersect the bounding box of `geom` in `gdf2`
        pos_idx = list(gdf1.sindex.intersection(geom.bounds))

        # Select the subset of `gdf1` based on these indices
        pos_gdf = gdf1.iloc[pos_idx]

        # Filter the subset to retain only geometries that precisely intersect with `geom`
        pre_gdf = pos_gdf[pos_gdf.intersects(geom)]

        if poly_id is not None and len(pre_gdf) > 0:
            # If `poly_id` is provided, assign the geometry ID from `gdf2` to the `geom_id` column in `pre_gdf`
            gid = gdf2[gdf2.geometry == geom][poly_id].iloc[0]
            pre_gdf["geom_id"] = gid

        # Concatenate the precise intersecting geometries to the final intersected DataFrame
        int_gdf = pd.concat([int_gdf, pre_gdf])
        counts.append(len(pre_gdf))  # Store the number of intersecting geometries for the current feature in gdf2

    gdf2["counts"] = counts  # Add the counts of intersecting geometries as a new column in gdf2
    return int_gdf  # Return the intersected subset of gdf1


def poverlay(
    gdf1: gpd.GeoDataFrame, gdf2: gpd.GeoDataFrame, how: str = "intersection", keep_geom_type: bool = False
) -> gpd.GeoDataFrame:
    """
    Performs a spatial overlay operation between two GeoDataFrames in parallel using multiple CPU cores.

    This function divides the first GeoDataFrame into chunks according to the number of available CPU cores
    and applies the specified overlay operation (e.g., intersection, union, difference) in parallel on each chunk
    with respect to the second GeoDataFrame. The results are then concatenated and returned as a single GeoDataFrame.

    Parameters
    ----------
    gdf1 : gpd.GeoDataFrame
        The first GeoDataFrame to be used in the spatial overlay operation.
    gdf2 : gpd.GeoDataFrame
        The second GeoDataFrame to be used in the spatial overlay operation.
    how : str, optional
        The type of overlay operation to perform. Options include "intersection", "union", "difference",
        "symmetric_difference", and "identity". Defaults to "intersection".
    keep_geom_type : bool, optional
        Whether to retain the original geometry type (e.g., Polygon, LineString) in the resulting overlay.
        If set to True, only features of the same geometry type are retained. Defaults to False.

    Returns
    -------
    gpd.GeoDataFrame
        A new GeoDataFrame resulting from the spatial overlay operation, with the same coordinate reference system
        (CRS) as the first input GeoDataFrame (`gdf1`).

    Examples
    --------
    >>> gdf1 = gpd.GeoDataFrame({"geometry": [Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])]})
    >>> gdf2 = gpd.GeoDataFrame({"geometry": [Polygon([(1, 1), (3, 1), (3, 3), (1, 3)])]})
    >>> result_gdf = poverlay(gdf1, gdf2, how="intersection")
    >>> print(result_gdf)
                                                 geometry
    0  POLYGON ((2.00000 1.00000, 2.00000 2.00000, 1....

    Notes
    -----
    - The spatial overlay operation is performed using the `geopandas.overlay` function. The parallelization is achieved
      using the `multiprocessing` library to divide and distribute the overlay operations across multiple CPU cores.
    - Ensure that both GeoDataFrames (`gdf1` and `gdf2`) have the same coordinate reference system (CRS) before applying
      the overlay operation to avoid unexpected results.

    Raises
    ------
    ValueError
        If the `how` parameter is not one of the supported overlay operation types: "intersection", "union",
        "difference", "symmetric_difference", or "identity".
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


def geocoding_google(address_or_zipcode: str, api_key: str) -> pd.Series:
    """
    Returns geographic coordinates (latitude and longitude) for a given address or zip code using the Google Geocoding API.

    This function utilizes the Google Geocoding API to convert a given address or zip code into geographic coordinates.
    The function returns the latitude and longitude as a pandas Series. If the request is unsuccessful or the address
    is not found, the function returns a Series with `(None, None)`.

    Parameters
    ----------
    address_or_zipcode : str
        A text-based address or zip code that needs to be geocoded.
    api_key : str
        A valid Google Maps API key required to access the Google Geocoding service.

    Returns
    -------
    pd.Series
        A pandas Series containing the latitude and longitude as floats. If the request fails or the address is not found,
        returns a Series with `(None, None)`.

    Examples
    --------
    >>> df[["lat", "lon"]] = df.apply(lambda row: geocoding_google(row.address, "your_api_key"), axis=1)
    >>> result = geocoding_google("1600 Amphitheatre Parkway, Mountain View, CA", "your_api_key")
    >>> print(result)
    lat    37.4224764
    lon   -122.0842499
    dtype: float64

    Notes
    -----
    - Make sure to enable the Google Geocoding API in your Google Cloud Console and provide a valid API key.
    - The API might return ambiguous results if the input address is incomplete or vague.
    - Consider handling `None` values in the returned Series if the API fails to find the address or the request limit is exceeded.

    Raises
    ------
    Exception
        If there is an error in the API request or response parsing, an exception is raised with an error message.
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


def reverse_geocoding_google(lat: float, lon: float, api_key: str) -> str:
    """
    Returns the postal code for a given geographic coordinate (latitude, longitude) using the Google Geocoding API.

    This function makes a reverse geocoding request to the Google Geocoding API to obtain the postal code associated
    with the provided latitude and longitude. If the postal code is found, it is returned as a string. If not,
    `None` is returned.

    Parameters
    ----------
    lat : float
        The latitude of the location to reverse geocode.
    lon : float
        The longitude of the location to reverse geocode.
    api_key : str
        A valid Google Maps API key for accessing the geocoding service.

    Returns
    -------
    str
        The postal code corresponding to the input geographic coordinates, if found. Returns `None` if no postal code
        is found or if the request fails.

    Examples
    --------
    >>> reverse_geocoding_google(37.4224764, -122.0842499, "your_api_key")
    '94043'

    >>> df["postcode"] = df.apply(lambda row: reverse_geocoding_google(row.lat, row.lon, "your_api_key"), axis=1)
    """
    lat = 0 if abs(lat) < 0.0001 else lat  # Prevent invalid 'latlng' error for very small values.
    lon = 0 if abs(lon) < 0.0001 else lon

    # Make the reverse geocoding request
    url = f"https://maps.googleapis.com/maps/api/geocode/json?latlng={lat},{lon}&key={api_key}"
    response = requests.get(url)
    data = response.json()

    # Parse the response to extract the postal code
    if "results" in data and len(data["results"]) > 0:
        for item in data["results"]:
            for component in item.get("address_components", []):
                if "postal_code" in component.get("types", []) and len(component["types"]) == 1:
                    return component.get("long_name", None)
    return None
