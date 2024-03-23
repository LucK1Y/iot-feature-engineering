from pathlib import Path
import numpy as np
import pandas as pd
import sys
from typing import Iterator, Tuple
import zipfile
from tqdm import tqdm
import re
import os
import concurrent.futures
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import pickle

MAX_WORKERS = 7
if MAX_WORKERS > os.cpu_count():
    print(f"You only have {os.cpu_count()} workers available.")
    MAX_WORKERS = os.cpu_count()


repo_base_path = Path(".").resolve()
assert str(repo_base_path).endswith(
    "csg_is"
), f"{repo_base_path} is not a valid path to the CSG_IS repository"
sys.path.append(str(repo_base_path))
from py_dataset import get_all_files_df

# from py_dataset import read_in_files

# from py_dataset import feature_selection
# from py_dataset import net_feature_engineering
# from py_dataset import entropy_feature_engineering
# from py_dataset.classes import DataSource
from py_dataset import sys_func
import argparse


all_Devices = {
    "Heqing_device1": "rp4",
    "Heqing_device2": "rp4",
    "Jing_device1": "rp3",
    "Jing_device2": "rp3",
    "Xi_sensor_1": "rp3",
    "Xi_sensor_2": "rp3",
    "Zien_device1": "rp3",
    "Zien_device2": "rp3",
}


def filter_single_device_single_data_source(
    df: pd.DataFrame, device: str, data_source: str
) -> pd.DataFrame:
    single_dev = df[df["device"] == device]
    assert (
        len(single_dev["device"].unique()) == 1
        and single_dev["device"].unique()[0] == device
    )

    single_dev_single_data_source = single_dev[single_dev["data_source"] == data_source]
    assert len(single_dev_single_data_source["data_source"].unique()) == 1

    print(single_dev_single_data_source.shape)
    # single_dev_single_data_source.head(1)

    return single_dev_single_data_source


def yield_log_files_from_zip(zip_file_path) -> Iterator[Tuple[str, str]]:
    with zipfile.ZipFile(zip_file_path, "r") as zip_file:
        file_infos = [
            file_info
            for file_info in zip_file.infolist()
            if file_info.filename.endswith(".log")
            and not (
                "__MACOSX" in file_info.filename and "SYS/._" in file_info.filename
            )
        ]
        for file_info in tqdm(
            file_infos, desc="Reading log files from zip", unit="files"
        ):
            try:
                with zip_file.open(file_info) as file:
                    yield file_info.filename, file.read().decode("utf-8")
            except Exception as ex:
                print(
                    f"cannot open: {zip_file_path}-{file_info.filename}. Reason: {ex}"
                )


def process_log_file(content: str):
    system_calls = re.findall(r"(?<=\s)(\w+)(?=\(arg0)", content)
    system_calls_string = " ".join(system_calls)

    return system_calls_string


def process_file(row, target_path_sys_calls):
    assert row["filetype"] == "zip", "Only zip files are supported"
    logs = []
    csv_file_name = f"{row['file_name']}_{row['label']}_logs.csv"
    path = target_path_sys_calls / csv_file_name
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
                "label": row["label"],
            }
        )

    logs_df = pd.DataFrame(logs)
    logs_df.to_csv(str(path), index=False)


def log_zips_to_csvs(target_path_sys_calls, data_path, DEVICE):
    df = get_all_files_df.main(data_path)
    df = filter_single_device_single_data_source(df, DEVICE, "SYS_data")
    print("files: ", df["file_name"].value_counts())

    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [
            executor.submit(process_file, row, target_path_sys_calls)
            for _, row in df.iterrows()
        ]
        for future in concurrent.futures.as_completed(futures):
            future.result()


def tokenizer(doc):
    # Using default pattern from CountVectorizer
    token_pattern = re.compile("(?u)\\b\\w\\w+\\b")
    return [t for t in token_pattern.findall(doc)]


def yield_dataframe_file(csv_files):
    # for csv_file in tqdm(csv_files, desc="Reading dataframes from csvs", unit="files"):
    for csv_file in csv_files:
        print(f"Reading {csv_file}")
        df = pd.read_csv(csv_file)

        yield df


def process_df(df):
    documents = df["system_calls"].to_numpy()
    vocab_df = set()
    for doc in documents:
        vocabs = set(tokenizer(str(doc)))
        vocab_df.update(vocabs)

    print(f"Processed {len(documents)} documents. Returning {len(vocab_df)} vocabs")
    return vocab_df


def create_vocab(csv_files):
    vocab_set = set()
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [
            executor.submit(process_df, df) for df in yield_dataframe_file(csv_files)
        ]
        for future in concurrent.futures.as_completed(futures):
            vocab_set.update(future.result())
            print(f"Vocab updated to: {vocab_set}")

    vocabulary = tuple(sorted(vocab_set))

    return vocabulary


def vectorize(df, csv_file, vectorizer, label_encoder):
    len_bf = len(df["system_calls"])
    df = df[df["system_calls"].notna()]
    df = df[df["label"].notna()]
    print(
        f"Removed {len_bf} - {len(df)} = {len_bf - len(df)} NaNs from documents in df: ",
        csv_file,
    )
    # print("uniqute labels:")
    # print(df["label"].unique())

    if len(df) == 0:
        print("Skipping empty dataframe: ", csv_file)
        return np.array([]), np.array([])

    documents = df["system_calls"].to_numpy()
    X_docs = vectorizer.transform(documents)
    X_docs = X_docs.toarray()

    labels = label_encoder.transform(df["label"])
    cols = np.column_stack((df["timestamp"].to_numpy(), labels))

    return X_docs, cols


def vectorize_all(csv_files, vocabulary, vectorizer, label_encoder):
    X = np.array([[]]).reshape(0, len(vocabulary))
    Z = np.array([[]]).reshape(0, 2)

    with tqdm(total=len(csv_files)) as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [
                executor.submit(vectorize, df, csv_file, vectorizer, label_encoder)
                for df, csv_file in zip(yield_dataframe_file(csv_files), csv_files)
            ]
            for future in concurrent.futures.as_completed(futures):
                X_docs, cols = future.result()
                X = np.concatenate([X, X_docs], axis=0)
                Z = np.concatenate([Z, cols], axis=0)
                pbar.update(1)

    assert (
        X.shape[0] == Z.shape[0]
    ), f"X.shape[0] = {X.shape[0]}!= Z.shape[0] = {Z.shape[0]}"

    return X, Z


def apply_tf_idf(X, vectorizer):
    transformer = TfidfTransformer()
    X_tf = transformer.fit_transform(X).toarray()

    print(X_tf.shape, X.shape)
    feature_names = vectorizer.get_feature_names_out()
    idf_weights = transformer.idf_
    top_features = np.argsort(idf_weights)[::-1][:10]  # get top 10 features
    print(feature_names[top_features])

    return X_tf


def parse_args():
    parser = argparse.ArgumentParser(description="", fromfile_prefix_chars="@")
    parser.add_argument(
        "-d",
        "--data_path_directory",
        type=Path,
        default=Path(
            "/media/<User>/DC/MAP_CreationOfNewDatasetsForDFL/code&data/0_raw_collected_data/"
        ),
    )
    parser.add_argument(
        "-v",
        "--directory_vocab_file",
        type=Path,
        default=Path(
            "/media/<User>/DC/IS_Data_Exploration_and_Feature_Engineering_for_an_IoT_Device_Behavior_Fingerprinting_Dataset/sys_system_calls_Xi_sensor_1"
        ),
    )
    parser.add_argument(
        "-x",
        "--device",
        type=str,
        default="Xi_sensor_1",
    )

    parser.add_argument("--test", action="store_true")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    DEVICE = args.device
    if not all_Devices.get(DEVICE):
        raise Exception("Cannot find Device: ", DEVICE)
    else:
        print("Using Device:", DEVICE, " Model: ", all_Devices.get(DEVICE))

    data_path = args.data_path_directory
    assert data_path.exists()

    directory_vocab_file = args.directory_vocab_file
    assert directory_vocab_file.exists()

    target_path_sys_calls: Path = directory_vocab_file / "sys_calls"
    target_path_sys_calls.mkdir(exist_ok=True)
    assert target_path_sys_calls.exists()

    output_file: Path = directory_vocab_file / "merged_data.npz"

    # 1. Create Csv files from log collection
    csv_files = [csv_file for csv_file in target_path_sys_calls.glob("**/*.csv")]
    if args.test:
        print("Csv_files: ", csv_files)
        print(data_path)
        print(directory_vocab_file)
        print(target_path_sys_calls)
    elif len(csv_files) < 8:  # we can assume for every label one csv file or more
        print("Not all Csvs found. Will start process zip to csvs.")
        log_zips_to_csvs(
            target_path_sys_calls=target_path_sys_calls,
            data_path=data_path,
            DEVICE=DEVICE,
        )
    else:
        print(f"CSVs already exist: {csv_files}\n at {target_path_sys_calls}")

    # 2. Create Vocab (collect all unique system call names)
    vocab_file = directory_vocab_file / "vocabulary.pkl"
    if vocab_file.exists():
        with open(str(vocab_file), "rb") as f:
            vocabulary = pickle.load(f)
        print("Loaded vocab:", vocabulary)
    elif args.test:
        print(f"Vocab file supposed to be stored at: {vocab_file} does not exist yet.")
    else:
        vocabulary = create_vocab(csv_files)
        with open(str(vocab_file), "wb") as f:
            pickle.dump(vocabulary, f)
    assert len(vocabulary) > 10, f"Vocabulary is milicious at {vocab_file}"

    # 3. Vectorize/encode all the Csv files
    label_encoder = sys_func.create_label_encoder_for_labels()
    vectorizer = CountVectorizer(vocabulary=vocabulary)
    if args.test or output_file.exists():
        print(f"output_file exists: {output_file.exists()}")
        print("Stopping.")
        return

    X, Z = vectorize_all(
        csv_files=csv_files,
        vocabulary=vocabulary,
        vectorizer=vectorizer,
        label_encoder=label_encoder,
    )
    print(f"Vectors shape: X: {X.shape} ; Z: {Z.shape}")

    # 4. Apply Tf-idf and store the results
    X_tf = apply_tf_idf(X, vectorizer=vectorizer)
    merged_array = np.column_stack([X_tf, Z])
    np.savez_compressed(str(output_file).rstrip(".npz"), merged_array)


if __name__ == "__main__":
    main()
