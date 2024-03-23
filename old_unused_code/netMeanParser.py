from typing import List

import pandas as pd


def filter_out_ip_addresses(df: pd.DataFrame, ip_address: List[str]) -> pd.DataFrame:
    return df[~((df["SourceIP"].isin(ip_address)) | (df["DestIP"].isin(ip_address)))]


def create_mean_net_df(df: pd.DataFrame, time_window, start_timestamp: pd.Timestamp) -> pd.DataFrame:
    series = []
    for timestamp, group_df in df.resample(time_window, origin=start_timestamp):
        protocol_counts = group_df["Protocol"].value_counts()
        features = {
            "timestamp": timestamp,
            "PacketCount": group_df.shape[0],
            "TotalLength": group_df["Length"].sum(),
            "AverageLength": group_df["Length"].mean(),
            "MedianLength": group_df["Length"].median(),
            "MinLength": group_df["Length"].min(),
            "MaxLength": group_df["Length"].max(),
            "VarianceLength": group_df["Length"].var(),
            "DifferentSourcePorts": group_df["SourcePort"].nunique(),
            "DifferentDestPorts": group_df["DestPort"].nunique(),
            "TcpPacketCount": protocol_counts.get("TCP", 0),
            "UdpPacketCount": protocol_counts.get("UDP", 0),
        }
        series.append(features)

    return pd.DataFrame(series).set_index("timestamp", inplace=False, drop=True)
