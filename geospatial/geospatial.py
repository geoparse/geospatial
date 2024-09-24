import math
from datetime import timedelta
from math import atan2, cos, radians, sin, sqrt
from multiprocessing import Pool, cpu_count

import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
import requests
from shapely.geometry import MultiPolygon, Point, Polygon
from shapely.ops import transform

#pd.options.mode.chained_assignment = None  # default='warn'


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


def overlay_parallel(gdf1, gdf2, how="intersection", keep_geom_type=False):
    n_cores = cpu_count()
    gdf1_chunks = np.array_split(gdf1, n_cores)
    gdf2_chunks = [gdf2] * n_cores
    inputs = zip(gdf1_chunks, gdf2_chunks, [how] * n_cores, [keep_geom_type] * n_cores)

    with Pool(n_cores) as pool:  # Create a multiprocessing pool and apply the overlay function in parallel on each chunk
        df = pd.concat(pool.starmap(gpd.overlay, inputs))
     return gpd.GeoDataFrame(df, crs=gdf1.crs)


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


def explode_line_to_points(row):
    points = [Point(x) for x in list(row["geometry"].coords)]  # create list of Point objects
    gdf = gpd.GeoDataFrame(
        index=range(len(points)), columns=row.index
    )  # create new GeoDataFrame with all columns and Point geometry
    gdf.loc[:, "geometry"] = points
    gdf.loc[:, row.index.drop("geometry")] = row[row.index.drop("geometry")].values
    return gdf


