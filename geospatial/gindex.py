def geom_to_cell(geoms: List[BaseGeometry], cell_type: str, res: int, dump: bool = False) -> Union[List[str], None]:
    """
    Converts a list of geometries into a set of unique spatial cells based on the specified cell type and resolution.

    This function takes a list of Shapely geometries (e.g., Polygon, MultiPolygon) and converts them into spatial cells
    using one of the supported cell systems: Geohash, S2, or H3. The resulting cells are returned as a list of unique
    cell IDs. If `dump` is set to True, the cells are saved to a file instead of being returned.

    Parameters
    ----------
    geoms : list of shapely.geometry.base.BaseGeometry
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
    """


def geom_to_cell_parallel(
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

    """


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

    """

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
    """

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
    """

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
    """
