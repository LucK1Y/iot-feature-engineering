from pathlib import Path
import numpy as np
import pandas as pd
import pickle
import sys
import argparse

repo_base_path = Path("./").resolve()
assert str(repo_base_path).endswith(
    "csg_is"
), f"{repo_base_path} is not a valid path to the CSG_IS repository"


COLUMNS_SELECTION_FILE_FROM_RP4 = (
    repo_base_path
    / "training"
    / "single_layer"
    / "other_time_windows"
    / "weight_selected_30s.csv"
)


def check_label_distribution(df):
    from collections import Counter
    import matplotlib.pyplot as plt

    all_labels = [label for label in df["label"].to_numpy()]

    label_counter = Counter(all_labels)

    # Print label distribution
    for label, count in label_counter.items():
        print(f"Label {label}: {count} samples")

    # Optional: Visualize the distribution
    labels, counts = zip(*label_counter.items())
    plt.bar(labels, counts)
    plt.xlabel("Label")
    plt.ylabel("Frequency")
    plt.title("Label Distribution")
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description="", fromfile_prefix_chars="@")
    parser.add_argument(
        "-d",
        "--dataset_input",
        type=Path,
        default=repo_base_path / "training" / "data" / "all_df_30s_Heqing_device2.csv",
    )
    parser.add_argument(
        "-v",
        "--dataset_output",
        type=Path,
        default=repo_base_path / "training" / "data" / "cleaned",
    )

    parser.add_argument("--test", action="store_true")

    args = parser.parse_args()
    return args


def create_dataset(DATASET_IN, dataset_for_fedstellar, test):

    print("[+] reading in ", DATASET_IN)
    print("[+] Creating ", dataset_for_fedstellar)

    sys.path.append(str(repo_base_path))
    from py_dataset import feature_plotting
    from py_dataset import sys_func

    df = pd.read_csv(str(DATASET_IN))
    print(f"Setting {df.columns[0]} as index.")
    df.set_index(df.columns[0], inplace=True)

    # 0. Select subset
    df.dropna(axis=0, how="any", inplace=True)
    df.reset_index(inplace=True, drop=True)

    selected_columns = pd.read_csv(COLUMNS_SELECTION_FILE_FROM_RP4)["0"].values
    selected_columns_that_are_in_rp3_dataset = list(
        set(selected_columns) & set(df.columns)
    )
    print("Subset: ", df[selected_columns_that_are_in_rp3_dataset].shape)

    check_label_distribution(df)
    assert set(selected_columns) == set(
        selected_columns
    ), f"this is a rp4 dataset => all should be in columns: \n{set(selected_columns) - set(selected_columns)}"

    subset = df[selected_columns_that_are_in_rp3_dataset].copy()
    subset["label"] = df["label"]

    df = subset

    # 1. Remove null values (created through merging)
    # old_df = df.copy()
    df.dropna(axis=0, how="any", inplace=True)
    df.reset_index(inplace=True, drop=True)

    # 2. encode Labels
    label_encoder = sys_func.create_label_encoder_for_labels()
    df["label"] = label_encoder.transform(df["label"])
    check_label_distribution(df)

    # 3. test dataset
    df = feature_plotting.test_transform_all_to_numeric_columns(df)

    # 4. Normalize data
    cols_to_exclude_from_scaling = ["label"]
    df = feature_plotting.minmax_scale_features(
        df,
        remove_outliers_iqr_all_columns=False,
        cols_to_exclude_from_scaling=cols_to_exclude_from_scaling,
    )
    print("Checking labels: ", df["label"].unique())
    check_label_distribution(df)

    # 5. Shuffle Data
    df = df.sample(frac=1).reset_index(drop=True)

    file_path = dataset_for_fedstellar / f"cleaned_{DATASET_IN.name}"
    file_path.parent.mkdir(exist_ok=True, parents=True)
    print(f"Store {df.shape} at {file_path}. First row:\n{df.head(1)}")

    if not test:
        df.to_csv(str(file_path), index=False)


def main():
    args = parse_args()

    DATASET_IN: Path = args.dataset_input
    dataset_for_fedstellar = args.dataset_output
    assert DATASET_IN.exists(), f"Dataset {DATASET_IN} does not exist"

    if DATASET_IN.is_file() and DATASET_IN.suffix[1:] == "csv":
        create_dataset(
            DATASET_IN=DATASET_IN,
            dataset_for_fedstellar=dataset_for_fedstellar,
            test=args.test,
        )

    elif DATASET_IN.is_dir():
        for file in DATASET_IN.glob("*.csv"):
            create_dataset(
                DATASET_IN=file,
                dataset_for_fedstellar=dataset_for_fedstellar,
                test=args.test,
            )

    else:
        raise ValueError("Cannot interpret: ", DATASET_IN)


if __name__ == "__main__":
    main()
