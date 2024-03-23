import pandas as pd
from sklearn.preprocessing import minmax_scale
import numpy as np


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


def _remove_outliers_iqr_all_columns(df):
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[~((df < lower_bound) | (df > upper_bound)).any(axis=1)]


def scale_sparse_matrix_rows(s, lowval=0, highval=1):
    d = s.data

    lens = s.getnnz(axis=1)
    idx = np.r_[0, lens[:-1].cumsum()]

    maxs = np.maximum.reduceat(d, idx)
    mins = np.minimum.reduceat(d, idx)

    minsr = np.repeat(mins, lens)
    maxsr = np.repeat(maxs, lens)

    D = highval - lowval
    scaled_01_vals = (d - minsr) / (maxsr - minsr)
    d[:] = scaled_01_vals * D + lowval

    return s


def minmax_scale_features(
    df, remove_outliers_iqr_all_columns=False, cols_to_exclude_from_scaling=None
):
    df_copy = df.copy()

    if cols_to_exclude_from_scaling and any(cols_to_exclude_from_scaling):
        exclude_from_scaling = df_copy[cols_to_exclude_from_scaling]
        df_copy.drop(columns=cols_to_exclude_from_scaling, inplace=True)

    if remove_outliers_iqr_all_columns:
        df_copy = _remove_outliers_iqr_all_columns(df_copy)

    df_copy = pd.DataFrame(minmax_scale(df_copy, axis=0), columns=df_copy.columns)
    df_copy = pd.concat([df_copy, exclude_from_scaling], axis=1)

    return df_copy


def test_transform_all_to_numeric_columns(df: pd.DataFrame, cols_to_exclude=None):
    non_numeric_cols = df.select_dtypes(exclude=["number"]).columns.tolist()
    if cols_to_exclude:
        for col in cols_to_exclude:
            if col in non_numeric_cols:
                non_numeric_cols.remove(col)

    for col in non_numeric_cols:
        df[col] = pd.to_numeric(df[col])

    return df
