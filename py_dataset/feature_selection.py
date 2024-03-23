from sklearn.feature_selection import (
    VarianceThreshold,
    chi2,
    f_classif,
    mutual_info_classif,
)
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
from py_dataset import feature_plotting
import numpy as np


def get_low_variance_features(df, threshold=0.0):
    vt = VarianceThreshold(threshold=threshold)
    vt.fit(df)
    return df.columns[~vt.get_support()].tolist()


def get_low_variance_features_for_sparse(sparse_matrix, threshold=0.0):
    variances = np.var(sparse_matrix.data, axis=0)

    vt = VarianceThreshold(threshold=threshold)
    vt.fit(sparse_matrix)
    return sparse_matrix.columns[~vt.get_support()].tolist()


def calculate_scores(df, labels):
    return pd.DataFrame(
        {
            "chi2": chi2(df, labels)[0],
            "f_classif": f_classif(df, labels)[0],
            "mutual_info_classif": mutual_info_classif(df, labels),
            "feature": df.columns,
        }
    )


def sort_by_mean(scores):
    scores["mean_score"] = scores[["chi2", "f_classif", "mutual_info_classif"]].mean(
        axis=1
    )
    scores.sort_values(by="mean_score", ascending=False, inplace=True)


def plot_single_scores(scores):
    scores = feature_plotting.minmax_scale_features(
        scores,
        remove_outliers_iqr_all_columns=False,
        cols_to_exclude_from_scaling=["feature"],
    )
    sort_by_mean(scores)

    plt.figure(figsize=(10, 6))
    plt.plot(scores["feature"], scores["chi2"], label="chi2")
    plt.plot(scores["feature"], scores["f_classif"], label="f_classif")
    plt.plot(
        scores["feature"], scores["mutual_info_classif"], label="mutual_info_classif"
    )
    plt.xlabel("Feature")
    plt.ylabel("Value")
    plt.title("Feature Importance")
    plt.legend()
    plt.show()


def calculate_scores_by_label(df, labels, columns=None):
    all_scores = []
    for label in labels.unique():
        print(f"Label: {label}")
        scores = pd.DataFrame(
            {
                "chi2": chi2(df, labels == label)[0],
                "f_classif": f_classif(df, labels == label)[0],
                "mutual_info_classif": mutual_info_classif(df, labels == label),
                "feature": df.columns if columns is None else columns,
                "label": label,
            }
        )
        all_scores.append(scores)

    df = pd.concat(all_scores)
    df.reset_index(inplace=True, drop=True)
    return df


def heatmap_scores_by_label_versus_features(scores):
    scores = feature_plotting.minmax_scale_features(
        scores,
        remove_outliers_iqr_all_columns=False,
        cols_to_exclude_from_scaling=["feature", "label"],
    )
    sort_by_mean(scores)

    pivot_df = scores.pivot(
        index="feature", columns="label", values="mutual_info_classif"
    )
    plt.figure(figsize=(16, 8))
    sns.heatmap(pivot_df, annot=True, cmap="coolwarm", linewidths=0.5)
    plt.xlabel("Labels")
    plt.ylabel("Features")
    plt.title("Mutual Information Scores")
    plt.show()

    pivot_df = scores.pivot(index="feature", columns="label", values="chi2")
    plt.figure(figsize=(16, 8))
    sns.heatmap(pivot_df, annot=True, cmap="coolwarm", linewidths=0.5)
    plt.xlabel("Labels")
    plt.ylabel("Features")
    plt.title("Chi 2 Scores")
    plt.show()

    pivot_df = scores.pivot(index="feature", columns="label", values="f_classif")
    plt.figure(figsize=(16, 8))
    sns.heatmap(pivot_df, annot=True, cmap="coolwarm", linewidths=0.5)
    plt.xlabel("Labels")
    plt.ylabel("Features")
    plt.title("F Classif Scores")
    plt.show()


def heatmap_for_single_feature_by_label(scores, feature: str):
    scores = feature_plotting.minmax_scale_features(
        scores,
        remove_outliers_iqr_all_columns=False,
        cols_to_exclude_from_scaling=["feature", "label"],
    )
    sort_by_mean(scores)

    first_feature_scores = scores[scores["feature"] == feature]
    melted_scores = pd.melt(
        first_feature_scores,
        id_vars=["label"],
        value_vars=["chi2", "f_classif", "mutual_info_classif"],
        var_name="score_type",
        value_name="score_value",
    )
    pivot_df = melted_scores.pivot(
        index="score_type", columns="label", values="score_value"
    )

    plt.figure(figsize=(16, 8))
    sns.heatmap(pivot_df, annot=True, cmap="coolwarm", linewidths=0.5)
    plt.xlabel("Labels")
    plt.ylabel("Score Types")
    plt.title(f"Scores for {feature}")
    plt.show()


def plot_scores_by_label(scores):
    scores = feature_plotting.minmax_scale_features(
        scores,
        remove_outliers_iqr_all_columns=False,
        cols_to_exclude_from_scaling=["feature", "label"],
    )

    fig, axs = plt.subplots(9, 1, figsize=(16, 40))

    for i, label in enumerate(scores["label"].unique()):
        axs[i].plot(
            scores[scores["label"] == label]["feature"],
            scores[scores["label"] == label]["chi2"],
            label=f"chi2_{label}",
        )
        axs[i].plot(
            scores[scores["label"] == label]["feature"],
            scores[scores["label"] == label]["f_classif"],
            label=f"f_classif_{label}",
        )
        axs[i].plot(
            scores[scores["label"] == label]["feature"],
            scores[scores["label"] == label]["mutual_info_classif"],
            label=f"mutual_info_classif_{label}",
        )
        axs[i].set_title(f"Label: {label}")
        axs[i].legend()

    plt.xlabel("Feature")
    plt.ylabel("Value")
    plt.title("Feature Importance")
    plt.show()


def plot_scores_by_label_single_plot_single_feature(scores, feature):
    scores = scores.copy()
    scores = scores[scores["feature"] == feature]
    scores.reset_index(inplace=True, drop=True)

    scores = feature_plotting.minmax_scale_features(
        scores,
        remove_outliers_iqr_all_columns=False,
        cols_to_exclude_from_scaling=["feature", "label"],
    )

    plt.figure(figsize=(10, 6))
    plt.plot(scores["label"], scores["chi2"], label="chi2")
    plt.plot(scores["label"], scores["f_classif"], label="f_classif")
    plt.plot(
        scores["label"], scores["mutual_info_classif"], label="mutual_info_classif"
    )
    plt.xlabel("Label")
    plt.ylabel("Value")
    plt.title(f"Feature Importance for {feature}")
    plt.legend()
    plt.show()


def plot_scores_by_label_single_plot(scores):
    global subset

    scores = feature_plotting.minmax_scale_features(
        scores.copy(),
        remove_outliers_iqr_all_columns=False,
        cols_to_exclude_from_scaling=["feature", "label"],
    )
    sort_by_mean(scores)
    scores.set_index("feature", inplace=True, drop=False)

    plt.figure(figsize=(10, 6))

    for label in scores["label"].unique():
        print(label)
        subset = scores[scores["label"] == label].copy()
        subset.reset_index(inplace=True, drop=True)

        # plt.plot(scores['feature'], scores['chi2'], label='chi2')
        # plt.plot(scores['feature'], scores['f_classif'], label='f_classif')
        # plt.plot(scores['feature'], scores['mutual_info_classif'], label='mutual_info_classif')
        plt.plot(
            subset.index,
            subset["chi2"],
            label=f"chi2_{label}",
        )
        plt.plot(
            subset.index,
            subset["f_classif"],
            label=f"f_classif_{label}",
        )
        plt.plot(
            subset.index,
            subset["mutual_info_classif"],
            label=f"mutual_info_classif_{label}",
        )

    plt.xlabel("Feature")
    # plt.xticks(scores.index, scores["feature"].unique())
    plt.xticks(range(len(subset)), subset["feature"])

    plt.ylabel("Value")
    plt.title("Feature Importance")
    plt.legend()
    plt.show()
