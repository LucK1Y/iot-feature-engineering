import pandas as pd
import numpy as np


def create_mean_net_df_single_label(
    df: pd.DataFrame, time_window="40s", start="start"
) -> pd.DataFrame:
    series = []
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df.index)
    for timestamp, group_df in df.resample(time_window, origin=start):
        if group_df.empty:
            features = {
                "timestamp": timestamp,
                "(OLD) PacketCount": 0,
                "(OLD) TotalLength": 0,
                "(OLD) AverageLength": 0,
                "(OLD) MedianLength": 0,
                "(OLD) MinLength": 0,
                "(OLD) MaxLength": 0,
                "(OLD) VarianceLength": 0,
                "(OLD) DifferentSourcePorts": 0,
                "(OLD) DifferentDestPorts": 0,
                "TcpPacketCount": 0,
                "UdpPacketCount": 0,
                "TcpUdpProtocolRatio": 0,
                "MeanInterPacketInterval": 0,
                "VarianceInterPacketInterval": 0,
                "MinInterPacketInterval": 0,
                "MaxInterPacketInterval": 0,
                "FirstDerivativeInterPacketInterval": 0,
                "SecondDerivativeInterPacketInterval": 0,
                "AverageBandwidth": 0,
                "VarianceBandwidth": 0,
                "MinBandwidth": 0,
                "MaxBandwidth": 0,
                "DifferentDestIPs": 0,
            }
            series.append(features)
            continue

        group_df = group_df.copy()
        protocol_counts = group_df["Protocol"].value_counts()

        group_df["InterPacketInterval"] = (
            group_df["timestamp"].diff().dt.total_seconds()
        )
        # group_df['InterPacketInterval'] = group_df['InterPacketInterval'].fillna(0)
        # group_df.fillna({"InterPacketInterval": 0}, inplace=True)

        group_df["BytesPerSecond"] = group_df["Length"].div(
            group_df["InterPacketInterval"]
        )
        # group_df.fillna({"BytesPerSecond": 0}, inplace=True)
        group_df.replace({"BytesPerSecond": np.inf}, 0, inplace=True)

        if group_df["InterPacketInterval"].isna().any():
            x = group_df

        features = {
            "timestamp": timestamp,
            "(OLD) PacketCount": group_df.shape[0],
            "(OLD) TotalLength": group_df["Length"].sum(),
            "(OLD) AverageLength": group_df["Length"].mean(),
            "(OLD) MedianLength": group_df["Length"].median(),
            "(OLD) MinLength": group_df["Length"].min(),
            "(OLD) MaxLength": group_df["Length"].max(),
            "(OLD) VarianceLength": group_df["Length"].var(),
            "(OLD) DifferentSourcePorts": group_df["SourcePort"].nunique(),
            "(OLD) DifferentDestPorts": group_df["DestPort"].nunique(),
            "TcpPacketCount": protocol_counts.get("TCP", 0),
            "UdpPacketCount": protocol_counts.get("UDP", 0),
            "TcpUdpProtocolRatio": protocol_counts.get("TCP", 0)
            / protocol_counts.get("UDP", 0.1),
            "MeanInterPacketInterval": group_df["InterPacketInterval"].mean(
                skipna=True
            ),
            "VarianceInterPacketInterval": group_df["InterPacketInterval"].var(
                skipna=True
            ),
            "MinInterPacketInterval": group_df["InterPacketInterval"].min(skipna=True),
            "MaxInterPacketInterval": group_df["InterPacketInterval"].max(skipna=True),
            "FirstDerivativeInterPacketInterval": group_df["InterPacketInterval"]
            .diff()
            .mean(skipna=True),
            "SecondDerivativeInterPacketInterval": group_df["InterPacketInterval"]
            .diff()
            .diff()
            .mean(skipna=True),
            "AverageBandwidth": group_df["BytesPerSecond"].mean(skipna=True),
            "VarianceBandwidth": group_df["BytesPerSecond"].var(skipna=True),
            "MinBandwidth": group_df["BytesPerSecond"].min(skipna=True),
            "MaxBandwidth": group_df["BytesPerSecond"].max(skipna=True),
            "DifferentDestIPs": group_df["DestIP"].nunique(),
        }
        series.append(features)

    return pd.DataFrame(series).set_index("timestamp", inplace=False, drop=True)


def create_mean_net_df_multiple_labels(
    df: pd.DataFrame, time_window="40s"
) -> pd.DataFrame:
    vectors = []
    for label in df["label"].unique():
        mean_df = create_mean_net_df_single_label(
            df[df["label"] == label], time_window=time_window
        )
        mean_df["label"] = label
        vectors.append(mean_df)

    return pd.concat(vectors)
