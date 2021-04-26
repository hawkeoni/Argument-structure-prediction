import json
from pathlib import Path
from collections import defaultdict

import typer
import pandas as pd
from tqdm import tqdm


def transform_jsonl(file: Path, outfile: Path):
    f = open(outfile, "w")
    label_mapping = {"entailment": "Evidence", "not_entailment": "Neutral"}
    for line in tqdm(open(file)):
        d = json.loads(line)
        d["sentence1"] = d["premise"]
        d["sentence2"] = d["hypothesis"]
        d["gold_label"] = label_mapping[d["label"]]
        f.write(json.dumps(d, ensure_ascii=False))
        f.write("\n")
    f.close()





def main(terra_folder: Path):
    files = [terra_folder / "train.jsonl", terra_folder / "val.jsonl"]
    for file in files:
        transform_jsonl(file, Path.cwd() / "datasets" / ("terra_" + file.name))


if __name__ == "__main__":
    typer.run(main)
