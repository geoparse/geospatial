import pandas as pd
import tqdm
from multiprocess import (  # The 'multiprocessing' module has a major limitation when it comes to IPython use. 'multiprocess' is a fork of the 'multiprocessing' module which uses dill instead of pickle to serialization and overcomes this issue conveniently.
    Pool,
    cpu_count,
)


def parallelize(function, df):
    ncores = cpu_count()
    chunk_size = len(df) // ncores
    chunks = [df.iloc[i : i + chunk_size] for i in range(0, len(df), chunk_size)]

    with Pool(ncores) as pool:
        df = pd.concat(tqdm.tqdm(pool.imap(function, chunks), total=ncores))
    # df = pd.concat(pool.map(function, chunks))
    return df
