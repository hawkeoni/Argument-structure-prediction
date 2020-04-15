import os
import sys
import json
from typing import List, Tuple, Dict, Any


def read_file(filename: str) -> Dict[str, Any]:
    return json.load(open(filename, encoding="utf8"))


def main(araucaria_folder):
    s = set()
    files = os.listdir(araucaria_folder)
    files = list(filter(lambda x: x.endswith(".json"), files))
    for file in files:
        filename = os.path.join(araucaria_folder, file)
        d = read_file(filename)
        for node_dict in d["nodes"]:
            s.add(node_dict["type"])
    print(s)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python process_araucaria.py araucaria_folder")
    else:
        main(sys.argv[1])