from typing import Optional

import pandas as pd
import requests
from shapely.geometry import LineString, Polygon



def way_to_geom(way_id: int, url: str):
    """
    Converts an OSM way ID into a Shapely Polygon or LineString object.

    This function retrieves the geometry corresponding to the given OSM way ID and
    returns it as a Shapely `Polygon` or `LineString` object based on whether the way
    forms a closed loop or not.

    Parameters
    ----------
    way_id : int
        The OpenStreetMap (OSM) way ID to be retrieved.
    url : str
        The URL endpoint for the Overpass API to request the geometry.

    Returns
    -------
    shapely.geometry.Polygon or shapely.geometry.LineString
        A Shapely `Polygon` object if the way forms a closed loop, or a `LineString`
        object otherwise.

    Notes
    -----
    - The function constructs an Overpass API query using the given way ID,
      requests the geometry, and then converts it into a Shapely geometry.
    - Assumes that the Overpass API returns data in JSON format with a "geometry" attribute.

    Examples
    --------
    >>> way_id = 123456
    >>> url = "https://overpass-api.de/api/interpreter"
    >>> geometry = way_to_geom(way_id, url)
    >>> print(geometry)
    POLYGON ((13.3888 52.5170, 13.3976 52.5291, 13.4286 52.5232, 13.3888 52.5170))
    """
    query = f"[out:json][timeout:600][maxsize:4073741824];way({way_id});out geom;"
    response = requests.get(url, params={"data": query}).json()
    response = response["elements"][0]
    geom = response["geometry"]
    coords = [(node["lon"], node["lat"]) for node in geom]
    if geom[0] == geom[-1]:  # Check if the way forms a closed loop
        return Polygon(coords)
    else:
        return LineString(coords)


def ways_to_geom(ids, url):
    """
    Converts an array of OpenStreetMap (OSM) way IDs into Shapely geometries.

    This function retrieves the geometries corresponding to the given OSM way IDs and
    returns a list of Shapely `LineString` or `Polygon` objects based on the geometries
    fetched from the OSM API.

    Parameters
    ----------
    ids : list of int
        A list of OSM way IDs to be retrieved.
    url : str
        The URL endpoint for the Overpass API to request the geometries.

    Returns
    -------
    list of shapely.geometry.LineString or shapely.geometry.Polygon
        A list of Shapely `LineString` or `Polygon` objects representing the geometries
        of the OSM ways. If the way forms a closed loop, it is returned as a `Polygon`;
        otherwise, it is returned as a `LineString`.

    Notes
    -----
    - The function constructs an Overpass API query using the given IDs, requests the
      geometries, and then converts them into Shapely geometries.
    - The function assumes that the Overpass API returns data in JSON format and expects
      the "geometry" attribute to contain the coordinates.

    Examples
    --------
    >>> way_ids = [123456, 234567, 345678]
    >>> url = "https://overpass-api.de/api/interpreter"
    >>> geometries = ways_to_geom(way_ids, url)
    >>> print(geometries)
    [<shapely.geometry.polygon.Polygon object at 0x...>,
     <shapely.geometry.linestring.LineString object at 0x...>]
    """
    query = "[out:json][timeout:600][maxsize:4073741824];"
    for item in ids:
        query += f"way({item});out geom;"

    response = requests.get(url, params={"data": query}).json()
    response = response["elements"]
    nodes = response[0]["geometry"]  # used later to determine if the way is a Polygon or a LineString
    ways = [item["geometry"] for item in response]

    geoms = []
    for way in ways:
        coords = [(node["lon"], node["lat"]) for node in way]
        if nodes[0] == nodes[-1]:  # in polygons the first and last items are the same
            geoms.append(Polygon(coords))
        else:
            geoms.append(LineString(coords))
    return geoms


def decode(encoded: str) -> list:
    """
    Decodes an encoded polyline string from Valhalla into a list of coordinates.

    Valhalla routing, map-matching, and elevation services use an encoded polyline format
    to store a series of latitude and longitude coordinates as a single string. This function
    decodes the polyline into a list of coordinates with six decimal precision.

    Parameters
    ----------
    encoded : str
        An encoded polyline string as per the Valhalla encoding format.

    Returns
    -------
    list of list of float
        A list of [longitude, latitude] pairs decoded from the input polyline string.

    Notes
    -----
    - The function uses six decimal degrees of precision for decoding Valhalla's encoded polylines.
    - The decoded coordinates are returned in [longitude, latitude] format.

    References
    ----------
    - https://github.com/valhalla/valhalla-docs/blob/master/decoding.md#decode-a-route-shape

    Examples
    --------
    >>> encoded_polyline = "_p~iF~ps|U_ulLnnqC_mqNvxq`@"
    >>> decoded_coords = decode(encoded_polyline)
    >>> print(decoded_coords)
    [[-120.2, 38.5], [-120.95, 40.7], [-126.453, 43.252]]
    """
    inv = 1.0 / 1e6  # Six decimal places of precision in Valhalla
    decoded = []
    previous = [0, 0]
    i = 0
    while i < len(encoded):  # For each byte in the encoded string
        ll = [0, 0]  # To store latitude and longitude
        for j in [0, 1]:
            shift = 0
            byte = 0x20
            while byte >= 0x20:  # Keep decoding bytes until the complete coordinate is read
                byte = ord(encoded[i]) - 63
                i += 1
                ll[j] |= (byte & 0x1F) << shift
                shift += 5
            ll[j] = previous[j] + (~(ll[j] >> 1) if ll[j] & 1 else (ll[j] >> 1))
            previous[j] = ll[j]
        # Convert to float and format the result
        decoded.append([float("%.6f" % (ll[1] * inv)), float("%.6f" % (ll[0] * inv))])
    return decoded


def map_matching(df: pd.DataFrame, cost: str, url: str, format: str = "osrm") -> Optional[dict]:
    """
    Performs map matching using Valhalla's Meili service.

    Map matching aligns a series of GPS points onto a road network. This function takes a DataFrame
    of coordinates, sends a request to the Meili map-matching service, and returns the matched
    coordinates along with other route information.

    Parameters
    ----------
    df : pd.DataFrame
        A pandas DataFrame containing the GPS coordinates to be map-matched. It should be in the
        format of [{"lon": float, "lat": float}, ...].
    cost : str
        The routing profile to use for map matching. Common values include "auto", "bicycle",
        or "pedestrian".
    url : str
        The URL endpoint for the Meili map-matching service.
    format : str, optional
        The response format for the request, either "osrm" or "geojson". Defaults to "osrm".

    Returns
    -------
    Optional[dict]
        A dictionary representing the JSON response from the map-matching service if the request
        is successful, otherwise None.

    Examples
    --------
    >>> coordinates = [{"lon": -73.9857, "lat": 40.7484}, {"lon": -73.9851, "lat": 40.7478}]
    >>> df = pd.DataFrame(coordinates)
    >>> url = "https://valhalla.mapzen.com/trace_attributes"
    >>> matched_route = map_matching(df, "auto", url)
    >>> print(matched_route)
    {'shape': '_p~iF~ps|U_ulLnnqC_mqNvxq`@', 'confidence_score': 1.0}
    """
    meili_head = '{"shape":'  # Initial portion of the request body
    meili_coordinates = df.to_json(orient="records")  # Convert DataFrame to JSON format

    meili_tail = f', "search_radius":150, "shape_match":"map_snap", "costing":"{cost}", "format":"{format}"}}'

    # Combine the header, coordinates, and tail into a single request body
    meili_request_body = meili_head + meili_coordinates + meili_tail

    # Send the request to the Meili service
    response = requests.post(url, data=meili_request_body, headers={"Content-type": "application/json"})
    if response.status_code == 200:
        return response.json()  # Convert the JSON response to a dictionary
    else:
        return None
