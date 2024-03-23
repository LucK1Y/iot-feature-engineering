from pathlib import Path
import numpy as np
import pickle
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import LabelEncoder

data_path = Path(
    "/media/<User>/DC/IS_Data_Exploration_and_Feature_Engineering_for_an_IoT_Device_Behavior_Fingerprinting_Dataset/sys_system_calls_Heqing_device2/"
)
default_vocab_file = data_path / "vocabulary.pkl"


def create_CountVectorizer(vocab_file=default_vocab_file) -> CountVectorizer:
    assert data_path.exists(), f"Data path does not exist: {data_path}"

    if vocab_file.exists():
        with open(str(vocab_file), "rb") as f:
            vocabulary = pickle.load(f)

        print("Loaded set:", vocabulary)

    return CountVectorizer(vocabulary=vocabulary)


def create_label_encoder_for_labels() -> LabelEncoder:
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

    return label_encoder
