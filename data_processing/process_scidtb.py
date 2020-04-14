import os
import sys
from typing import List, Tuple


def read_file(filename: str) -> List[Tuple[str, str]]:
    sentence_class_tuples: List[Tuple[str, str]] = []
    tokens: List[str] = []
    tags: List[str] = []
    for line in open(filename, encoding="utf8"):
        word, tag = line.strip().split("\t")
        word = word.replace("-RRB-", ")").replace("-LRB-", "(")
        if tag.startswith("B") and tokens:
            sentence = ' '.join(tokens)
            sentence_tag = "Evidence" if "support" in tags[-1] else "None"
            sentence_class_tuples.append((sentence, sentence_tag))
            tokens = []
            tags = []
        tokens.append(word)
        tags.append(tag)
    sentence = ' '.join(tokens)
    sentence_tag = "Evidence" if "support" in tags[-1] else "None"
    sentence_class_tuples.append((sentence, sentence_tag))
    return sentence_class_tuples


def main(scitdb_folder):
    n = 0
    f = open("scitdb_full.txt", "w", encoding="utf8")
    files = os.listdir(scitdb_folder)
    for file in files:
        filename = os.path.join(scitdb_folder, file)
        for sentence, sentence_class in read_file(filename):
            f.write(f"{sentence}\t{sentence_class}\n")
            n += 1
    print(f"Finished dataset processing, {n} sentences written.")



if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python process_scitdb.py scidtb_folder")
    else:
        main(sys.argv[1])