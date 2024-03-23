import re
from pathlib import Path

import pandas as pd


FILE_PATH_PATTERN = re.compile(
    r"0_raw_collected_data/(?P<researcher_name>\w+)/(?P<device_name>\w+)/(?P<label>\w+)/(?P<data_source>\w+)/(?P<file_name>[\w.-]+\.(?:csv|zip))$"
)


def _is_ds_stores_or_in_backup(file_path: Path):
    return file_path.match(".DS_Store") or file_path.match(".backup")


def _is_not_file(file_path: Path):
    return not file_path.is_file()


def _is_txt(file_path: Path):
    if file_path.suffix[1:] == "txt":
        print("txt file found, will drop", file_path)
        return True
    return False


def _get_file_size(file_path: Path):
    return file_path.stat().st_size


def _get_file_type(file_path: Path):
    if not file_path.suffix[1:]:
        print("Found a file without type: ", file_path, file_path.suffix)
        return "no_filetype"
    return file_path.suffix[1:]


def _get_groups(file_path: Path):
    match = FILE_PATH_PATTERN.search(str(file_path))
    if match:
        assert (
            len(match.groups()) == 5
        ), f"Matched groups are not 5, but: {len(match.groups())}"
        return match.groups()
    raise Exception("No match found for: ", file_path)


def main(data_path: Path):
    pattern_all_visible_files = "**/*.*"
    all_files = list(data_path.glob(pattern_all_visible_files))

    df = pd.DataFrame(all_files, columns=["file_path"])
    mask = df["file_path"].map(_is_not_file)
    df = df[~mask]
    mask = df["file_path"].map(_is_ds_stores_or_in_backup)
    df = df[~mask]
    mask = df["file_path"].map(_is_txt)
    df = df[~mask]
    print(df.head(1))

    df = df.reset_index(drop=True)
    print(df.index)

    df["filetype"] = df["file_path"].map(lambda a: _get_file_type(a))
    print(df["filetype"].value_counts())

    df["filesize_bytes"] = df["file_path"].map(lambda a: _get_file_size(a))
    print(df.head(1))

    dicts = df["file_path"].map(lambda a: _get_groups(a))
    dicts_df = pd.DataFrame(
        dicts.tolist(),
        columns=[
            "researcher_name",
            "device_name",
            "label",
            "data_source",
            "file_name",
        ],
    )
    df = pd.concat([df, dicts_df], axis=1)

    # df2 = df.copy(deep=True)
    df.replace(
        {
            "data_source": {
                "FLSYS": "FLS_data",
                "Block": "block_data",
                "KERN": "KERN_data",
                "NET": "network_data",
                "Network": "network_data",
                "ENTROPY": "entropy_data",
                "Entropy": "entropy_data",
                "BLOCK": "block_data",
                "RES": "RES_data",
            },
            "label": {
                "1_normal_4h": "1_normal",
                "3_thetick_4h": "3_thetick",
                "2_ransomware_4h": "2_ransomware",
                "5_httpbackcoor": "5_httpbackdoor",
            },
        },
        inplace=True,
    )

    df["device"] = df["researcher_name"] + "_" + df["device_name"]

    print(df.value_counts("data_source"))

    return df
