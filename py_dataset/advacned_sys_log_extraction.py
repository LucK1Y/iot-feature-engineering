import glob
import json
import marshal
import os
import pathlib
import re
from typing import AnyStr, List, Dict, Tuple

import pandas as pd

main_process_pattern = re.compile(
    r"\s+(?P<service_name>.+\d*)\s+\((?P<pid>\d+)\),\s+(?P<events>\d+)\s+events,\s+(?P<cpu_usage>\d+\.\d+)%"
)
system_calls_pattern = re.compile(
    r"\s*(?P<syscall>\S+)\s+(?P<calls>\d+)\s+(?P<total>\d+(\.\d+)?)\s+(?P<min>\d+(\.\d+)?)\s+(?P<avg>\d+(\.\d+)?)\s+(?P<max>\d+(\.\d+)?)\s+\s+(?P<stddev>\d+(\.\d+)?)%"
)

uptime_pattern = re.compile(r"UPTIME:(?P<uptime>\d+\.\d+)")


def get_header_data(text: AnyStr) -> Dict[AnyStr, AnyStr]:
    match = main_process_pattern.match(text)
    if match:
        return match.groupdict()
    else:
        return {}


def get_system_calls_process(text: AnyStr) -> List[Dict[AnyStr, AnyStr]]:
    matches = system_calls_pattern.finditer(text)
    if matches:
        return [match.groupdict() for match in matches]
    else:
        return []


def get_uptime(text: AnyStr) -> AnyStr:
    match = uptime_pattern.search(text)
    if match:
        return match.group("uptime")
    else:
        return ""


def parse(text: AnyStr) -> Tuple[List[Dict[AnyStr, AnyStr]], AnyStr]:
    _, _, summary_text = text.rpartition(" Summary of events:")
    processes = []
    uptime = ""
    for paragraph in summary_text.split("\n\n\n"):
        main_process_str, _, system_calls_str = paragraph.partition("------\n")

        main_process: Dict[AnyStr, AnyStr | List[Dict[AnyStr, AnyStr]]] = (
            get_header_data(main_process_str)
        )
        if not main_process:
            uptime = get_uptime(main_process_str)
            continue
        system_calls_process = get_system_calls_process(system_calls_str)

        main_process["system_calls"] = system_calls_process
        processes.append(main_process)
    return processes, uptime


def store_dict_as_json(output: AnyStr, data):
    json_dump = json.dumps(data)
    with open(output, "w") as json_file:
        json_file.write(json_dump)


def store_dict_as_marshall(output: AnyStr, data):
    with open(output, "wb") as marshal_file:
        marshal.dump(data, marshal_file)


def store_dict_as_csv(output: AnyStr, data):
    df = pd.DataFrame(data)
    df.to_csv(output, index=False)


def loop_sys_log_files(base_path: AnyStr):
    pattern_filepath = os.path.join(base_path, "**/*.log")
    data = []
    for file_path in glob.iglob(pattern_filepath, recursive=True):
        file_timestamp = pathlib.Path(file_path).stem

        # timestamp = pd.to_datetime(int(file_name), unit="s")

        with open(file_path, "r") as file:
            content = file.read()
            processes, file_uptime = parse(content)

        file = {
            "timestamp": file_timestamp,
            "processes": processes,
            "uptime": file_uptime,
        }
        data.append(file)

    return data


if __name__ == "__main__":
    path = "/media/<User>/DC/IS_Data_Exploration_and_Feature_Engineering_for_an_IoT_Device_Behavior_Fingerprinting_Dataset/0_raw_collected_dataZien_device1/"

    data = loop_sys_log_files(path)
    # store_dict_as_json("sys_data.json", data)
    # store_dict_as_marshall("sys_data.marshall", data)

    store_dict_as_csv("sys_data.csv", data)
