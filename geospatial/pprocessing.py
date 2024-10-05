from typing import Callable

import pandas as pd
import tqdm
from multiprocess import Pool, cpu_count


def parallelize(function: Callable[[pd.DataFrame], pd.DataFrame], df: pd.DataFrame) -> pd.DataFrame:
    """
    Parallelizes the execution of a function across multiple CPU cores for a given DataFrame.

    This function splits the input DataFrame into chunks based on the number of available CPU cores
    and applies the specified function in parallel to each chunk using the `multiprocess` library.
    It then concatenates the results back into a single DataFrame.

    The `multiprocessing` module has a major limitation when it comes to IPython use.
    `multiprocess` is a fork of the `multiprocessing` module which uses `dill` instead of `pickle`
    for serialization and overcomes this issue conveniently.

    Parameters
    ----------
    function : Callable[[pd.DataFrame], pd.DataFrame]
        A function that processes a chunk of the DataFrame and returns a transformed DataFrame.
    df : pd.DataFrame
        The DataFrame to be processed in parallel.

    Returns
    -------
    pd.DataFrame
        The resulting DataFrame after applying the function to all chunks in parallel.

    Examples
    --------
    >>> import pandas as pd
    >>> from multiprocess import Pool

    >>> def sample_function(chunk):
    ...     chunk["new_column"] = chunk["existing_column"] * 2
    ...     return chunk

    >>> # Create a sample DataFrame
    >>> input_df = pd.DataFrame({"existing_column": [1, 2, 3, 4]})

    >>> # Apply the function to the DataFrame in parallel
    >>> result_df = parallelize(sample_function, input_df)
    >>> print(result_df)
       existing_column  new_column
    0                1           2
    1                2           4
    2                3           6
    3                4           8

    Notes
    -----
    This function uses the `multiprocess` library instead of the standard `multiprocessing` module
    because `multiprocess` supports a broader range of serialization scenarios, making it more
    suitable for use with IPython and Jupyter notebooks.
    """
    # Get the number of available CPU cores for parallel processing
    ncores = cpu_count()

    # Determine the size of each chunk based on the number of cores
    chunk_size = len(df) // ncores

    # Split the DataFrame into chunks
    chunks = [df.iloc[i : i + chunk_size] for i in range(0, len(df), chunk_size)]

    # Create a multiprocessing pool and apply the function to each chunk in parallel
    with Pool(ncores) as pool:
        df = pd.concat(tqdm.tqdm(pool.imap(function, chunks), total=ncores))
        # df = pd.concat(pool.map(function, chunks))
    return df
