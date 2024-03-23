#!/usr/bin/env python
# coding: utf-8

# # Idea
# Read into all sys files of a given sensor and calc the TF-IDF Vector for each file.
# - Then Vocab as columns/features and each file as a row.
# - Then Analyse Columns as before

# In[1]:

import sys

sys.path.append("/home/<User>/repos/csg_is")

from pathlib import Path

import numpy as np
import pandas as pd


# In[2]:


from py_dataset import get_all_files_df
from py_dataset import read_in_files
from py_dataset import feature_plotting


# In[3]:


data_path = Path(
    "/media/<User>/DC/MAP_CreationOfNewDatasetsForDFL/code&data/0_raw_collected_data/"
)
assert data_path.exists()


# In[4]:


df = get_all_files_df.main(data_path)
df.head(1)


# # Get only DataSource = Sys and Device = Heqing_device2

# In[5]:


single_dev = df[df["device"] == "Heqing_device2"]
assert len(single_dev["device"].unique()) == 1


# In[6]:


single_dev_single_data_source = single_dev[single_dev["data_source"] == "SYS_data"]
assert len(single_dev_single_data_source["data_source"].unique()) == 1
single_dev_single_data_source.shape


# In[7]:


single_dev_single_data_source.head(1)


# In[8]:


single_dev_single_data_source["file_name"].value_counts()


# In[9]:


from typing import Iterator, Tuple
import zipfile
from tqdm import tqdm


def yield_log_files_from_zip(zip_file_path) -> Iterator[Tuple[str, str]]:
    with zipfile.ZipFile(zip_file_path, "r") as zip_file:
        file_infos = [
            file_info
            for file_info in zip_file.infolist()
            if file_info.filename.endswith(".log")
        ]
        for file_info in tqdm(
            file_infos, desc="Reading log files from zip", unit="files"
        ):
            with zip_file.open(file_info) as file:
                yield file_info.filename, file.read().decode("utf-8")


# In[10]:


import re


def process_log_file(content: str):
    system_calls = re.findall(r"(?<=\s)(\w+)(?=\(arg0)", content)
    system_calls_string = " ".join(system_calls)

    return system_calls_string


# In[11]:


import os
import concurrent.futures


def process_file(row):
    assert row["filetype"] == "zip", "Only zip files are supported"
    logs = []
    csv_file_name = f"{row['file_name']}_logs.csv"
    path = (
        Path(
            "/media/<User>/DC/IS_Data_Exploration_and_Feature_Engineering_for_an_IoT_Device_Behavior_Fingerprinting_Dataset/"
        )
        / csv_file_name
    )
    if path.exists():
        print(f"Skipping {csv_file_name}")
        return

    for file_name, content in yield_log_files_from_zip(row["file_path"]):
        base_name = os.path.basename(file_name)
        timestamp = os.path.splitext(base_name)[0]
        system_calls_string = process_log_file(content)

        logs.append(
            {
                "timestamp": timestamp,
                "system_calls": system_calls_string,
                "experiment": row["experiment"],
            }
        )

    logs_df = pd.DataFrame(logs)
    logs_df.to_csv(str(path), index=False)


def main():
    with concurrent.futures.ProcessPoolExecutor(max_workers=3) as executor:
        futures = [
            executor.submit(process_file, row)
            for _, row in single_dev_single_data_source.iterrows()
        ]
        for future in concurrent.futures.as_completed(futures):
            future.result()


if __name__ == "__main__":
    main()
