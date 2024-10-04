import requests
from shapely.geometry import LineString, Polygon

# pd.options.mode.chained_assignment = None  # default='warn'


def ways_to_geom(ids, url):
    """
    Convert an array of OpenStreetMap (OSM) way IDs into Shapely geometries.

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


def way_to_geom(way_id, url):
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
