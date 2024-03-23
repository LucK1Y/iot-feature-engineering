from collections import defaultdict
import pandas as pd


def create_mean_sys_df(
    df: pd.DataFrame, time_window, start_timestamp: pd.Timestamp
) -> pd.DataFrame:

    series = []
    for timestamp, group_df in df.resample(time_window, origin=start_timestamp):
        processes_store = defaultdict(dict)
        for index, process_array in group_df.iterrows():
            print(index)
            print(process_array)

            for process in process_array["processes"]:
                pid = process["pid"]
                processes_store[pid]["max_cpu_usage"] = max(
                    processes_store[pid].get("cpu_usage", 0),
                    float(process["cpu_usage"]),
                )
                processes_store[pid]["cpu_usage"] = processes_store[pid].get(
                    "cpu_usage", 0
                ) + float(process["cpu_usage"])
                processes_store[pid]["occurances"] = (
                    processes_store[pid].get("occurances", 0) + 1
                )
                processes_store[pid]["events"] = processes_store[pid].get(
                    "events", 0
                ) + int(process["events"])
                processes_store[pid]["system_calls"] = (
                    processes_store[pid].get("system_calls", "")
                    + " "
                    + " ".join([call["syscall"] for call in process["system_calls"]])
                )
                processes_store[pid]["service_name"] = process["service_name"]

        features = {"timestamp": timestamp, "processes": list(processes_store.values())}
        series.append(features)

    return pd.DataFrame(series).set_index("timestamp", inplace=False, drop=True)


# not exaclty sure if this runs, but see time_windowing/scratch_files/features_sys.ipynb
# for a working example
if __name__ == "__main__":
    import time
    import numpy as np
    import json

    with open("time_windowing/systemcalls/data/sys_data.json", "r") as file:
        content = json.loads(file.read())

    df = pd.DataFrame(content)

    def is_timestamp_in_microseconds(df: pd.DataFrame) -> bool:
        max_timestamp = df["timestamp"].max()
        return max_timestamp > time.time()

    df["timestamp"] = df["timestamp"].astype(float)

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

    mean_df = create_mean_sys_df(df, time_window="20s", start_timestamp=None)
    mean_df.to_csv("mean_df_test.csv")
