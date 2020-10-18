import json
from pathlib import Path
import typer

def main(araucaria_folder: Path):
    s = set()
    files = araucaria_folder.glob("*json")
    for file in files:
        d = json.loads(file.read_text())
        for node_dict in d["nodes"]:
            s.add(node_dict["type"])
    print(s)


if __name__ == "__main__":
    typer.run(main)
