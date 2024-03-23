import os

import pandas as pd

from time_windowing import constants, netMeanParser
from time_windowing import parsingFunctions
from time_windowing.classes import DataType


def setup_for_time_slicing_inplace(df: pd.DataFrame):
    parsingFunctions.clean_and_set_index_timestamp(df)


def create_window_sliced_depending_on_type(
    df: pd.DataFrame, start_timestamp: pd.Timestamp
) -> pd.DataFrame:
    data_type = DataType.from_columns(df.columns.tolist())
    print(f"Data type: {data_type}")

    match data_type:
        case (
            DataType.FileSystem
            | DataType.KernelEvents
            | DataType.ResourceUsageEvents
            | DataType.BlockInputOutputEvents
        ):
            return df.resample(constants.TIME_WINDOW, origin=start_timestamp).mean()
        case DataType.NetRecords:
            df = netMeanParser.filter_out_ip_addresses(
                df, ip_address=constants.IP_ADDRESSES_TO_REMOVE
            )
            return netMeanParser.create_mean_net_df(
                df, time_window=constants.TIME_WINDOW, start_timestamp=start_timestamp
            )
        case DataType.SystemCalls:
            raise NotImplementedError("SystemCalls not implemented yet")


if __name__ == "__main__":
    PATH = "/media/<User>/DC/IS_Data_Exploration_and_Feature_Engineering_for_an_IoT_Device_Behavior_Fingerprinting_Dataset/0_raw_collected_dataZien_device1/"
    pattern_filepath = os.path.join(PATH, constants.CSV_PATTERN_FILEPATH)

    df_collection = []
    smallest_timestamp = pd.Timestamp.max
    for file_name, df_file in parsingFunctions.yield_filename_and_dataframes_from_csv(
        pattern_filepath=pattern_filepath
    ):
        print(f"Found file: {file_name}")
        setup_for_time_slicing_inplace(df_file)

        if df_file.index[0] < smallest_timestamp:
            smallest_timestamp = df_file.index[0]
        df_collection.append(df_file)

    print(f"Smallest Timestamp: {smallest_timestamp}")
    df_mean_collection = []
    for df_file in df_collection:
        df_mean = create_window_sliced_depending_on_type(
            df=df_file, start_timestamp=smallest_timestamp
        )
        print(f"Shape: {df_mean.shape}")

        print(
            f"Start and End of Index Timestamp: {df_mean.index[0]} - {df_mean.index[-1]}"
        )
        df_mean_collection.append(df_mean)

    # all_df = pd.merge(df_mean_collection)
    all_df = pd.concat(df_mean_collection, axis=1)
    print(f"all_df shape: {all_df.shape}")
    all_df.to_csv("all_df.csv")
