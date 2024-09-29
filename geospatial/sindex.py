import collections
import json
from datetime import datetime
from math import sqrt
from multiprocessing import Pool, cpu_count
from time import time
from typing import List, Union

import geopandas as gpd
import numpy as np
from h3 import h3
from polygon_geohasher.polygon_geohasher import geohash_to_polygon, polygon_to_geohashes
from s2 import s2
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry

# pd.options.mode.chained_assignment = None  # default='warn'


# s2.polyfill() function covers the hole in a polygon too (which is not correct).
# geom_to_cell_parallel() function splits a polygon to smaller polygons without holes


def geom_to_cell(geoms: List[BaseGeometry], cell_type: str, res: int, dump: bool = False) -> Union[List[str], None]:
    """
    Converts a list of geometries into a set of unique spatial cells based on the specified cell type and resolution.

    This function takes a list of Shapely geometries (e.g., Polygon, MultiPolygon), and converts them into spatial cells
    using one of the supported cell systems: Geohash, S2, or H3. The resulting cells are returned as a list of unique
    cell IDs. If `dump` is set to True, the cells are saved to a file instead of being returned.

    Args:
        geoms (List[BaseGeometry]): A list of Shapely geometry objects (Polygon or MultiPolygon).
        cell_type (str): The type of spatial cell system to use. Supported values are "geohash", "s2", or "h3".
        res (int): The resolution level for the spatial cells. The resolution parameter determines the granularity of the cells.
        dump (bool, optional): If True, the cells are saved to a file on the desktop in a folder named after `cell_type`.
                               If False, the function returns a list of cell IDs. Default is False.

    Returns:
        Union[List[str], None]: If `dump` is False, a list of unique cell IDs is returned. If `dump` is True, None is returned
                                after saving the cells to a file.

    Raises:
        ValueError: If `cell_type` is not one of the supported values ("geohash", "s2", "h3").

    Example:
        ```
        # Define a list of Shapely geometries
        geometries = [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]), MultiPolygon([...])]

        # Convert geometries to H3 cells at resolution 9
        h3_cells = geom_to_cell(geometries, cell_type="h3", res=9)
        ```
    """
    polys = []
    for geom in geoms:
        if geom.geom_type == "Polygon":
            polys += [geom.__geo_interface__]
        elif geom.geom_type == "MultiPolygon":  # If MultiPolygon, extract each Polygon separately
            polys += [g.__geo_interface__ for g in geom.geoms]

    cells = []
    if cell_type == "geohash":
        cells = set()
        for geom in geoms:
            cells |= polygon_to_geohashes(geom, precision=res, inner=False)  # Collect Geohashes for each Polygon
        cells = list(cells)

    elif cell_type == "s2":
        for poly in polys:
            cells += s2.polyfill(poly, res, geo_json_conformant=True, with_id=True)  # Use S2 to fill each Polygon
        cells = [item["id"] for item in cells]  # Keep only the cell IDs
        cells = list(set(cells))  # Remove duplicates

    elif cell_type == "h3":
        for poly in polys:
            cells += h3.polyfill(poly, res, geo_json_conformant=True)  # Use H3 to fill each Polygon

    else:
        raise ValueError(f"Unsupported cell type: {cell_type}. Choose 'geohash', 's2', or 'h3'.")

    if dump:
        with open(f"~/Desktop/{cell_type}/{datetime.now()}.txt", "w") as json_file:
            json.dump(cells, json_file)
        return None
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


def h3_stats(geom, h3_res, compact=False):
    cells = geom_to_cell(geom, cell="h3", res=h3_res)
    area = h3.hex_area(h3_res, unit="km^2")
    if compact:
        cells = h3.compact(cells)
    return len(cells), area
