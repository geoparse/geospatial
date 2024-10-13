import collections
import json
from datetime import datetime
from math import sqrt
from multiprocessing import Pool, cpu_count
from time import time
from typing import List, Tuple, Union

import geopandas as gpd
import numpy as np
from h3 import h3
from polygon_geohasher.polygon_geohasher import geohash_to_polygon, polygon_to_geohashes
from s2 import s2
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry

# s2.polyfill() function covers the hole in a polygon too (which is not correct).
# ppoly_cell() function splits a polygon to smaller polygons without holes


def poly_cell(geoms: List[Union[Polygon, MultiPolygon]], cell_type: str, res: int, dump: bool = False) -> Union[List[str], None]:
    """
    Converts a list of geometries into a set of unique spatial cells based on the specified cell type and resolution.

    This function takes a list of Shapely geometries (e.g., Polygon, MultiPolygon) and converts them into spatial cells
    using one of the supported cell systems: Geohash, S2, or H3. The resulting cells are returned as a list of unique
    cell IDs. If `dump` is set to True, the cells are saved to a file instead of being returned.

    Parameters
    ----------
    geoms : list of shapely.geometry.Polygon or shapely.geometry.MultiPolygon
        A list of Shapely geometry objects (Polygon or MultiPolygon).
    cell_type : str
        The type of spatial cell system to use. Supported values are "geohash", "s2", or "h3".
    res : int
        The resolution level for the spatial cells. The resolution parameter determines the granularity of the cells.
    dump : bool, optional
        If True, the cells are saved to a file on the desktop in a folder named after `cell_type`.
        If False, the function returns a list of cell IDs. Default is False.

    Returns
    -------
    list of str or None
        If `dump` is False, a list of unique cell IDs is returned.
        If `dump` is True, None is returned after saving the cells to a file.

    Raises
    ------
    ValueError
        If `cell_type` is not one of the supported values ("geohash", "s2", "h3").

    Examples
    --------
    >>> from shapely.geometry import Polygon, MultiPolygon
    >>> geometries = [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]), MultiPolygon([...])]
    >>> # Convert geometries to H3 cells at resolution 9
    >>> h3_cells = poly_cell(geometries, cell_type="h3", res=9)
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


def ppoly_cell(
    mdf: gpd.GeoDataFrame, cell_type: str, res: int, compact: bool = False, verbose: bool = False
) -> Tuple[List[str], int]:
    """
    Performs a parallelised conversion of geometries in a GeoDataFrame to cell identifiers of a specified type
    (e.g., Geohash, S2, or H3), optionally compacting the result to reduce the number of cells.

    This function first divides the bounding box of the input GeoDataFrame into smaller grid cells, then calculates
    the intersection between these grid cells and the input geometries. The resulting geometries are processed in
    parallel to generate cell identifiers according to the specified `cell_type` and `res` (resolution). The result
    can be compacted to reduce the number of cells.

    Parameters
    ----------
    mdf : gpd.GeoDataFrame
        A GeoDataFrame containing geometries that need to be converted to cell identifiers.

    cell_type : str
        The type of cell identifier to use. Options are:
        - "geohash": Converts geometries to Geohash identifiers.
        - "s2": Converts geometries to S2 cell tokens.
        - "h3": Converts geometries to H3 cell tokens.

    res : int
        The resolution or precision level of the cell identifiers. Higher values indicate finer precision.

    compact : bool, optional, default=False
        If True, compact the resulting cells to reduce their number. This is typically applicable for S2 and H3 cells.

    verbose : bool, optional, default=False
        If True, print timing and progress information to the console.

    Returns
    -------
    Tuple[List[str], int]
        - A list of cell identifiers as strings, corresponding to the geometries in the input GeoDataFrame.
        - The total number of unique cell identifiers.

    Raises
    ------
    ValueError
        If an invalid `cell_type` is provided. Supported types are "geohash", "s2", and "h3".

    Example
    -------
    >>> # Assuming `mdf` is a GeoDataFrame with geometries:
    >>> cells, count = ppoly_cell(mdf, cell_type="s2", res=10, compact=True, verbose=True)
    >>> print(f"Generated {count} cells: {cells}")
    """
    if verbose:
        print(datetime.now())
        print("\nSlicing the bounding box of polygons ... ", end="")
        start_time = time()

    # Determine the number of slices and grid cells based on CPU cores
    n_cores = cpu_count()
    slices = 128 * n_cores

    # Calculate the bounding box dimensions
    minlon, minlat, maxlon, maxlat = mdf.total_bounds
    dlon = maxlon - minlon
    dlat = maxlat - minlat
    ratio = dlon / dlat

    # Calculate the number of grid cells in x and y directions
    x_cells = round(sqrt(slices) * ratio)
    y_cells = round(sqrt(slices) / ratio)

    # Calculate step size for grid cells
    steplon = dlon / x_cells
    steplat = dlat / y_cells

    # Create grid polygons based on bounding box slices
    grid_polygons = []
    for lat in np.arange(minlat, maxlat, steplat):
        for lon in np.arange(minlon, maxlon, steplon):
            llon, llat, ulon, ulat = (lon, lat, lon + steplon, lat + steplat)  # lower lat, upper lat
            polygon = Polygon([(llon, llat), (ulon, llat), (ulon, ulat), (llon, ulat)])
            grid_polygons.append(polygon)

    gmdf = gpd.GeoDataFrame(geometry=grid_polygons, crs=mdf.crs)  # Create a GeoDataFrame with grid polygons

    if verbose:
        elapsed_time = round(time() - start_time)
        print(f"{elapsed_time} seconds.   {slices} slices created.")

        print("Performing intersection between grid and polygons ... ", end="")
        start_time = time()

    # Perform intersection between input geometries and grid cells
    gmdf = gpd.overlay(mdf, gmdf, how="intersection")  # grid mdf

    if verbose:
        elapsed_time = round(time() - start_time)
        print(f"{elapsed_time} seconds.   {len(gmdf)} intersected slices.")

        print("Calculating cell IDs in parallel ... ", end="")
        start_time = time()

    # Shuffle geometries for even load distribution across chunks
    gmdf = gmdf.sample(frac=1)
    geom_chunks = np.array_split(list(gmdf.geometry), 4 * n_cores)
    inputs = zip(geom_chunks, [cell_type] * 4 * n_cores, [res] * 4 * n_cores)

    # Parallel processing to generate cells
    with Pool(n_cores) as pool:
        cells = pool.starmap(poly_cell, inputs)
    cells = [item for sublist in cells for item in sublist]  # Flatten the list of cells

    if verbose:
        elapsed_time = round(time() - start_time)
        print(f"{elapsed_time} seconds.")

    # Remove duplicates based on cell type
    if cell_type in {"geohash", "s2"}:
        if verbose:
            print("Removing duplicate cells ... ", end="")
            start_time = time()
        cells = list(set(cells))  # Remove duplicate cells
        if verbose:
            elapsed_time = round(time() - start_time)
            print(f"{elapsed_time} seconds.")

    cell_counts = len(cells)  # Total unique cell count

    # Compact the cells if needed
    if compact:
        if verbose:
            print("Compacting cells ... ", end="")
            start_time = time()
        cells = compact_cells(cells, cell_type)
        if verbose:
            elapsed_time = round(time() - start_time)
            print(f"{elapsed_time} seconds.")

    return cells, cell_counts


def poly_cell_parallel_2(mdf, cell_type, res, compact=False, verbose=False, dump=True):
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
        pool.starmap(poly_cell, inputs)
    return


def cell_to_geom(cells: list, cell_type: str) -> tuple:
    """
    Converts a list of spatial cells to their corresponding geometries and resolution levels.

    The function takes a list of spatial cells (e.g., Geohash, H3, or S2) and converts each cell
    into a geometry object (Polygon) based on the specified cell type. It also calculates the resolution
    level for each cell.

    Parameters
    ----------
    cells : list
        A list of spatial cells represented as strings. Each cell corresponds to a spatial area
        in a specific grid system (e.g., Geohash, H3, or S2).

    cell_type : str
        The type of spatial cell system used. Accepted values are:
        - "geohash" : Geohash spatial indexing system.
        - "h3"      : H3 hexagonal spatial indexing system.
        - "s2"      : S2 spherical spatial indexing system.

    Returns
    -------
    tuple
        A tuple containing:
        - `res` : list of int
            A list of resolution levels corresponding to each cell in the input.
        - `geoms` : list of shapely.geometry.Polygon
            A list of Polygon geometries representing the spatial boundaries of the input cells.

    Raises
    ------
    ValueError
        If `cell_type` is not one of "geohash", "h3", or "s2".
    """
    # Check for valid cell_type
    if cell_type not in {"geohash", "h3", "s2"}:
        raise ValueError(f"Invalid cell_type '{cell_type}'. Accepted values are: 'geohash', 'h3', 's2'.")

    # Determine resolution level based on cell type
    res = [
        len(cell)
        if cell_type == "geohash"
        else cell[1]
        if cell_type == "h3"
        else s2.CellId.from_token(cell).level()  # cell = token
        for cell in cells
    ]

    # Create geometry objects based on cell type
    geoms = [
        geohash_to_polygon(cell)
        if cell_type == "geohash"
        else Polygon(s2.s2_to_geo_boundary(cell, geo_json_conformant=True))
        if cell_type == "s2"
        else Polygon(h3.h3_to_geo_boundary(cell, geo_json=True))
        for cell in cells
    ]

    return res, geoms


def compact_cells(cells: list, cell_type: str) -> list:
    """
    Compacts a list of spatial cells (e.g., Geohash, S2, or H3) by merging adjacent cells into parent cells.

    The function takes a list of spatial cells and compacts them into larger cells if possible, reducing the total number
    of cells by merging adjacent cells into their parent cell at a coarser resolution. The compaction process differs based
    on the specified `cell_type` and its respective hierarchy.

    Parameters
    ----------
    cells : list
        A list of spatial cells represented as strings. Each cell corresponds to a spatial area in a specific grid system
        (e.g., Geohash, H3, or S2).

    cell_type : str
        The type of spatial cell system used. Accepted values are:
        - "geohash" : Geohash spatial indexing system.
        - "h3"      : H3 hexagonal spatial indexing system.
        - "s2"      : S2 spherical spatial indexing system.

    Returns
    -------
    list
        A list of compacted spatial cells. Each cell is represented as a string and is at the coarsest resolution possible
        based on the input cells.

    Raises
    ------
    ValueError
        If `cell_type` is not one of "geohash", "h3", or "s2".

    Notes
    -----
    - For `h3`, the function uses the built-in `h3.compact()` method.
    - For `s2`, the compaction merges cells up to their parent cells by considering the S2 hierarchy.
    - For `geohash`, cells are merged based on shared prefixes.
    """
    if cell_type == "h3":
        return list(h3.compact(cells))
    elif cell_type == "s2":
        # Convert S2 cell IDs from tokens
        cells = [s2.CellId.from_token(item) for item in cells]
        res = cells[0].level()  # Assuming all S2 cells have the same resolution
        num_children = 4
    elif cell_type == "geohash":
        res = len(cells[0])  # Resolution is based on the length of Geohash strings
        num_children = 32
    else:
        raise ValueError(f"Invalid cell_type '{cell_type}'. Accepted values are: 'geohash', 'h3', 's2'.")

    # Initialize list to store compacted cells
    compact_cells = []
    for i in range(res, 0, -1):
        # Get parent cell IDs based on the type
        parent_ids = [cell.parent() if cell_type == "s2" else cell[: i - 1] for cell in cells]
        count_dict = collections.Counter(parent_ids)  # Count occurrences of each parent cell

        # Get indices of parent cells with the required number of children
        idx = [i for i, item in enumerate(parent_ids) if count_dict.get(item, 0) == num_children]

        # Create a mask to exclude compacted cells
        mask = [True] * len(cells)
        for ix in idx:
            mask[ix] = False
        cells = [item for i, item in enumerate(cells) if mask[i]]

        # Append compacted cells to the result
        compact_cells += cells
        cells = list(set([item for item in parent_ids if count_dict.get(item, 0) == num_children]))

    # Include any remaining cells in the compacted list
    compact_cells += cells

    if cell_type == "geohash":
        return compact_cells
    else:  # Convert S2 cells back to tokens
        return [item.to_token() for item in compact_cells]


def uncompact_s2(compact_tokens: list, level: int) -> list:
    """
    Expands a list of compacted S2 cell tokens to a specified resolution level.

    This function takes a list of compact S2 cell tokens and generates their child cells up to the desired
    resolution level. It is used to "uncompact" S2 cells that have been previously compacted, producing a
    more detailed representation.

    Parameters
    ----------
    compact_tokens : list
        A list of S2 cell tokens represented as strings. These tokens are at a coarser resolution level and
        will be expanded into their child cells.

    level : int
        The target S2 cell resolution level to which the input tokens should be expanded. The resolution level
        determines the size of the child cells. A higher level corresponds to finer granularity (smaller cells).

    Returns
    -------
    list
        A list of S2 cell tokens represented as strings. Each token corresponds to a child cell of the input
        compact tokens, expanded to the specified resolution level.

    Raises
    ------
    ValueError
        If the provided `level` is less than or equal to the resolution level of the input `compact_tokens`.

    Example
    -------
    >>> compact_tokens = ["89c2847c", "89c2847d"]
    >>> uncompact_s2(compact_tokens, level=10)
    ["89c2847c1", "89c2847c2", "89c2847c3", ..., "89c2847d1", "89c2847d2", ...]
    """
    uncompact_tokens = []
    for token in compact_tokens:
        cell_id = s2.CellId.from_token(token)  # Convert each token to an S2 CellId object
        uncompact_tokens += list(cell_id.children(level))  # Generate child cells at the specified level
    # Convert each CellId object back to a token and remove duplicates
    uncompact_tokens = [item.to_token() for item in uncompact_tokens]
    return list(set(uncompact_tokens))


def h3_stats(geom: BaseGeometry, h3_res: int, compact: bool = False) -> Tuple[int, float]:
    """
    Computes H3 cell statistics for a given geometry at a specified resolution.

    This function takes a Shapely geometry object and computes the number of H3 cells covering the geometry at a
    specified resolution. It also calculates the area of each H3 cell at the given resolution. Optionally, the function
    can return the compacted set of H3 cells, reducing the number of cells required to represent the geometry.

    Parameters
    ----------
    geom : shapely.geometry.base.BaseGeometry
        A Shapely geometry object (e.g., Polygon or MultiPolygon) representing the area of interest.
    h3_res : int
        The H3 resolution level for generating spatial cells. The resolution level controls the granularity of the cells.
    compact : bool, optional
        If True, the function returns a compacted set of H3 cells, reducing the number of cells needed to represent the geometry.
        Default is False.

    Returns
    -------
    tuple
        A tuple containing:
        - int: Number of H3 cells covering the given geometry.
        - float: Area of each H3 cell at the specified resolution, in square kilometers.

    Examples
    --------
    >>> from shapely.geometry import Polygon
    >>> geom = Polygon([(-122.0, 37.0), (-122.0, 38.0), (-121.0, 38.0), (-121.0, 37.0), (-122.0, 37.0)])
    >>> h3_stats(geom, h3_res=9, compact=True)
    (512, 0.001)

    Notes
    -----
    The function utilizes the H3 library for generating and compacting H3 cells and for calculating cell area. The area
    is always returned in square kilometers ("km^2").
    """
    cells = poly_cell(geom, cell="h3", res=h3_res)
    area = h3.hex_area(h3_res, unit="km^2")
    if compact:
        cells = h3.compact(cells)
    return len(cells), area
