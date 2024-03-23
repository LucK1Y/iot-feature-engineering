import numpy as np
import pandas as pd
import glob
from typing import Tuple, Iterator
import pandas as pd
import glob
import time
import glob
import pandas as pd
import os


def is_timestamp_in_microseconds(df: pd.DataFrame) -> bool:
    max_timestamp = df["timestamp"].max()
    return max_timestamp > time.time()


def clean_and_set_index_timestamp(df: pd.DataFrame):
    if "timestamp" in df.columns:
        pass
    elif "time" in df.columns:
        df["timestamp"] = df["time"]
    elif "Time" in df.columns:
        df["timestamp"] = df["Time"]

    if is_timestamp_in_microseconds(df):
        # df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df["timestamp"] = pd.to_datetime(np.floor(df["timestamp"] / 1000), unit="s")
    else:
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")

    df.set_index("timestamp", inplace=True, drop=True)

    df.drop(columns=["time", "Time", "timestamp"], inplace=True, errors="ignore")
    df.sort_index(inplace=True)


def yield_filename_and_dataframes_from_csv(
    pattern_filepath,
) -> Iterator[Tuple[str, pd.DataFrame]]:
    for csv_file_paths in glob.iglob(pattern_filepath, recursive=True):
        yield os.path.basename(csv_file_paths), pd.read_csv(csv_file_paths)
