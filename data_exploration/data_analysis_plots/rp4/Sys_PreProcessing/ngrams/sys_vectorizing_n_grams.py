# %% [markdown]
#
# # Introduction
# Use vectorizers to read in data. See here: https://stackoverflow.com/questions/31784011/scikit-learn-fitting-data-into-chunks-vs-fitting-it-all-at-once
#
# We cannot fit the data all at the same time on the vectorizer, as it takes too much memory. Luckily, this is not needed. We first iterate over chunks of text data and build up the vocabulary of the corpus. Then we can use it to fit the CountVectorizer efficiently.
#
# Then we can go over the chunks of text data again and transform them with the CountVectorizer into vectors. We can easily store all vectors of the complete data in memory.
#

# %%
from pathlib import Path
import re
import pandas as pd
from tqdm import tqdm
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import pickle
from sklearn.preprocessing import LabelEncoder

import concurrent.futures
import pickle
import os
import itertools


# %%
# max CPUs to use
max_workers = 3

# %%
data_path = Path(
    "/media/<User>/DC/IS_Data_Exploration_and_Feature_Engineering_for_an_IoT_Device_Behavior_Fingerprinting_Dataset/sys_system_calls_Heqing_device2/"
)
assert data_path.exists()

csv_files = [csv_file for csv_file in data_path.glob("**/*.csv")]


# %%
def tokenizer(doc):
    # Using default pattern from CountVectorizer
    token_pattern = re.compile("(?u)\\b\\w\\w+\\b")
    return [t for t in token_pattern.findall(doc)]


def yield_dataframe_file():
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


# %% [markdown]
# ### Creating the Vocab for The CountVectorizer

# %%
vocab_file = data_path / "vocabulary.pkl"

if vocab_file.exists():
    with open(str(vocab_file), "rb") as f:
        vocabulary = pickle.load(f)
    print("Loaded vocab:", vocabulary)
else:
    vocab_set = set()

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_df, df) for df in yield_dataframe_file()]
        for future in concurrent.futures.as_completed(futures):
            vocab_set.update(future.result())

    vocabulary = tuple(sorted(vocab_set))
    with open(str(vocab_file), "wb") as f:
        pickle.dump(vocabulary, f)

    print("Saved vocab:", vocabulary)

# %%
# vocab_file = data_path / "vocabulary.pkl"

# if vocab_file.exists():
#     with open(str(vocab_file), 'rb') as f:
#         vocabulary = pickle.load(f)
#     print("Loaded vocab:", vocabulary)
# else:
#     vocab_set = set()

#     with concurrent.futures.ThreadPoolExecutor(max_workers = max_workers) as executor:
#         futures = [executor.submit(process_df, df) for df in yield_dataframe_file()]
#         results = [future.result() for future in futures]

#     for vocab_df in results:
#         vocab_set.update(vocab_df)

#     vocab = tuple(sorted(vocab_set))
#     with open(str(vocab_file), 'wb') as f:
#         pickle.dump(vocab, f)

# %% [markdown]
# #### Encoding the systemcalls


# %%
def build_ngrams(arr, n):
    pairs = list(itertools.combinations(arr, n))

    pairs = [" ".join(pair) for pair in pairs]

    return tuple(sorted(set(pairs)))


ngram_vocab = build_ngrams(vocabulary, 2)
print("Ngram vocab:", len(ngram_vocab))

vocabulary += ngram_vocab

print("Complete vocab:", len(vocabulary))


vocab_file_ngrams = data_path / "vocabulary_ngrams.pkl"
with open(str(vocab_file_ngrams), "wb") as f:
    pickle.dump(vocabulary, f)


# %%
vectorizer = CountVectorizer(ngram_range=(1, 2), vocabulary=vocabulary)
X = vectorizer.transform(["bind getresgid32 dup3"])

# %%
labels = [
    "1_normal",
    "2_ransomware",
    "3_thetick",
    "4_bashlite",
    "5_httpbackdoor",
    "6_beurk",
    "7_backdoor",
    "8_bdvl",
    "9_xmrig",
]
label_encoder = LabelEncoder()
label_encoder.fit(labels)


# %%
def process_df(df, csv_file):
    len_bf = len(df["system_calls"])
    df = df[df["system_calls"].notna()]
    print(
        f"Removed {len_bf} - {len(df)} = {len_bf - len(df)} NaNs from documents in df: ",
        csv_file,
    )

    if len(df) == 0:
        print("Skipping empty dataframe: ", csv_file)
        return np.array([]), np.array([])

    documents = df["system_calls"].to_numpy()
    X_docs = vectorizer.transform(documents)
    X_docs = X_docs.toarray()

    labels = label_encoder.transform(df["experiment"])
    cols = np.column_stack((df["timestamp"].to_numpy(), labels))

    return X_docs, cols


# %%
X = np.array([[]]).reshape(0, len(vocabulary))
Z = np.array([[]]).reshape(0, 2)


with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = [
        executor.submit(process_df, df, csv_file)
        for df, csv_file in zip(yield_dataframe_file(), csv_files)
    ]
    for future in concurrent.futures.as_completed(futures):
        X_docs, cols = future.result()
        X = np.concatenate([X, X_docs], axis=0)
        Z = np.concatenate([Z, cols], axis=0)


assert (
    X.shape[0] == Z.shape[0]
), f"X.shape[0] = {X.shape[0]}!= Z.shape[0] = {Z.shape[0]}"

# %%
# vectorizer = CountVectorizer(vocabulary=vocabulary)
# X = np.array([[]]).reshape(0, len(vocabulary))
# Z = np.array([[]]).reshape(0, 2)

# with tqdm(total=len(csv_files)) as pbar:
#     for i, df in enumerate(yield_dataframe_file()):
#         len_bf = len(df["system_calls"])
#         df = df[df["system_calls"].notna()]
#         print(f"Removed {len_bf} - {len(df)} = {len_bf - len(df)} NaNs from documents in df: ", csv_files[i])

#         documents = df["system_calls"].to_numpy()
#         if len(documents) == 0:
#             print("Skipping empty dataframe: ", csv_files[i])
#             continue

#         X_docs = vectorizer.transform(documents)
#         X = np.concatenate([X, X_docs.toarray()], axis=0)

#         labels = label_encoder.transform(df["experiment"])
#         cols = np.column_stack((df["timestamp"].to_numpy(), labels))
#         Z = np.concatenate([Z, cols], axis=0)

#         assert X.shape[0] == Z.shape[0], f"X.shape[0] = {X.shape[0]} != Z.shape[0] = {Z.shape[0]}"

#         pbar.update(1)


# %%
transformer = TfidfTransformer()
X_tf = transformer.fit_transform(X).toarray()

# %%
X_tf.shape, X.shape, Z.shape,

# %%
# from sklearn.pipeline import Pipeline
# from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

# pipeline = Pipeline([
#     ('vectorizer', CountVectorizer(vocabulary=vocabulary)),
#     ('tfidf', TfidfTransformer())
# ])

# # Fit the pipeline to the data and transform it
# X_transformed = pipeline.fit_transform(X)

# X_transformed_dense = X_transformed.toarray()

# %%
feature_names = vectorizer.get_feature_names_out()
idf_weights = transformer.idf_
top_features = np.argsort(idf_weights)[::-1][:10]  # get top 10 features
print(feature_names[top_features])

# %%
merged_array = np.column_stack([X_tf, Z])

output_file = str(data_path / "merged_data_ngrams.npz")
np.savez_compressed(output_file, merged_array)
