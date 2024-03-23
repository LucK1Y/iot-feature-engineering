import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from pathlib import Path


def resample_high_entropy_grouping_by_label_multi_label(
    df, time_window="10s", start=None
):
    # assert pd.api.types.is_datetime64_dtype(df.index.dtype), "Index is not datetime type"

    df = df.copy()
    df.sort_index(inplace=True)

    series = []

    for label, group_df in df.groupby(df["label"]):
        for timestamp, group_df in group_df.resample(time_window, origin=start):
            group_df = group_df[group_df["entropy"] >= 6]

            entropy_file_count = group_df["file_path"].nunique()
            series.append(
                {
                    "timestamp": timestamp,
                    "entropy_file_count": entropy_file_count,
                    "label": label,
                }
            )

    new_vectors = pd.DataFrame(series)
    new_vectors.set_index("timestamp", inplace=True, drop=True)
    return new_vectors


def resample_high_entropy_grouping_by_label_single_label(
    df, time_window="10s", start=None
):
    # assert pd.api.types.is_datetime64_dtype(df.index.dtype), "Index is not datetime type"

    df = df.copy()
    df.sort_index(inplace=True)

    series = []

    for timestamp, df in df.resample(time_window, origin=start):
        df = df[df["entropy"] >= 6]

        entropy_file_count = df["file_path"].nunique()
        series.append(
            {"timestamp": timestamp, "entropy_file_count": entropy_file_count}
        )

    new_vectors = pd.DataFrame(series)
    new_vectors.set_index("timestamp", inplace=True, drop=True)
    return new_vectors


def get_file_type(file_path: str):
    file_path = Path(file_path)
    if not file_path.suffix[1:]:
        print("Found a file without type: ", file_path, file_path.suffix)
        return "no_filetype"
    return file_path.suffix[1:]


def resample_high_entropy_grouping_by_label_and_filetype(df, time_window="10s"):
    """
    samples by filetype, as it was shown that different filetypes have different entropy distributions
    However not feasible to use all filetypes, as the number of filetypes is too high (some files do not have a disting filetype)
    e.g. Found a file without type:  /tmp/wdg_retries

    =>> do NOT use this function
    """

    df = df.copy()
    df.sort_index(inplace=True)
    df["file_type"] = df["file_path"].apply(get_file_type)

    series = []

    vocab = df["file_type"].unique()
    vectorizer = CountVectorizer(vocabulary=vocab)

    for label, group_df in df.groupby(df["label"]):
        for timestamp, group_df in group_df.resample(time_window):
            group_df = group_df[group_df["entropy"] >= 5]

            doc = " ".join(group_df["file_type"])
            series.append({"timestamp": timestamp, "doc": doc, "label": label})

    new_vectors = pd.DataFrame(series)
    new_vectors.set_index("timestamp", inplace=True, drop=True)

    X = vectorizer.transform(new_vectors["doc"])

    transformer = TfidfTransformer()
    X_tf_idf = transformer.fit_transform(X).toarray()
    new_vectors.drop(columns=["doc"], inplace=True)
    new_vectors = pd.concat(
        [
            new_vectors,
            pd.DataFrame(X_tf_idf, columns=vectorizer.get_feature_names_out()),
        ],
        axis=1,
    )

    return new_vectors
