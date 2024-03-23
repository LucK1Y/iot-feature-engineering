import os
import time
import zipfile
from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd


def _is_timestamp_in_microseconds(df_x: pd.DataFrame) -> bool:
    max_timestamp = df_x["timestamp"].max()
    return max_timestamp > time.time()


def _get_vector_detailed_title(row) -> str:
    return (
        row["researcher_name"]
        + "_"
        + row["device_name"]
        + "_"
        + row["label"]
        + "_"
        + row["feature"]
    )


def _get_group_vector_title(row) -> str:
    return (
        row["researcher_name"] + "_" + row["device_name"]
    )  # + "_" + row['experiment'] + "_" + row['feature']


def _clean_and_set_index_timestamp(df_x: pd.DataFrame):
    if "timestamp" in df_x.columns:
        pass
    elif "time" in df_x.columns:
        df_x["timestamp"] = df_x["time"]
    elif "Time" in df_x.columns:
        df_x["timestamp"] = df_x["Time"]

    df_all = df_x.copy()
    # is set to none if not numeric => then dropping the none values
    df_x["timestamp"] = pd.to_numeric(df_x["timestamp"], errors="coerce")

    tobe_dropped_indices = df_x.index[df_x["timestamp"].isnull()]
    if tobe_dropped_indices.any():
        print("! Dropped rows in DataFrames:", tobe_dropped_indices.shape)
        for index in tobe_dropped_indices:
            print(df_x.iloc[index])

        df_x.dropna(subset=["timestamp"], inplace=True)
        df_x.reset_index(inplace=True)

    if _is_timestamp_in_microseconds(df_x):
        # df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df_x["timestamp"] = pd.to_datetime(np.floor(df_x["timestamp"] / 1000), unit="s")
    else:
        df_x["timestamp"] = pd.to_datetime(df_x["timestamp"], unit="s")

    df_x.set_index("timestamp", inplace=True, drop=True)

    df_x.drop(columns=["time", "Time", "timestamp"], inplace=True, errors="ignore")
    df_x.sort_index(inplace=True)


def _extract_timestamps_log_files(zip_file_path) -> pd.DataFrame:
    log_files = []
    try:
        with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
            for file_name in zip_ref.namelist():
                if file_name.endswith(".log"):
                    base_name = os.path.basename(file_name)
                    file_name_without_extension = os.path.splitext(base_name)[0]
                    log_files.append(file_name_without_extension)
    except Exception as e:
        print("Error in file: ", zip_file_path)
        print(e)

    log_df = pd.DataFrame(log_files, columns=["timestamp"])
    return log_df


def get_all_columns(file_path: Path, filetype: str):
    if filetype == "csv":
        df = pd.read_csv(str(file_path))
    elif filetype == "zip":
        df = _extract_timestamps_log_files(file_path)
    else:
        raise Exception("Filetype not supported: ", filetype)

    try:
        _clean_and_set_index_timestamp(df)
    except Exception as e:
        print("Error in file: ", file_path)
        raise e

    return df


def get_all_timestamps(
    file_path: Path, filetype: str, resample_time_window=None
) -> Tuple[pd.Timestamp, pd.Timestamp, pd.DataFrame]:
    """
    Returns Min, Max, and DataFrame of the timestamps in the file
    """
    df = get_all_columns(file_path, filetype)

    if resample_time_window:
        df_sizes = df.resample(resample_time_window).size()
        df_sizes = df_sizes.to_frame(name="vector_count")
    else:
        df_sizes = df.filter(items=["timestamp"])
    return df_sizes.index.min(), df_sizes.index.max(), df_sizes


# get_all_columns_from_single_data_source_for_csv_files
def get_all_columns_from_single_data_source_for_csv_files(
    df: pd.DataFrame,
) -> pd.DataFrame:
    data_frames = []

    for _, row in df.iterrows():
        assert row["filetype"] == "csv", "Only csv files are supported"

        df = get_all_columns(row["file_path"], row["filetype"])
        df["label"] = row["label"]

        if "index" in df.columns:
            print("Dropping index column")
            df.drop(columns=["index"], inplace=True)

        data_frames.append(df)

    return pd.concat(data_frames)
