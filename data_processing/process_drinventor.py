from pathlib import Path
from typing import List, Tuple
from collections import defaultdict

import typer
import pandas as pd
from lxml import etree
from lxml.etree import _Element


def read_xml(filename: str):
    text = open(filename, "rb").read()
    return etree.fromstring(text)


def read_folder(folder: Path) -> Tuple[_Element, _Element]:
    files = folder.glob("*xml")
    files = list(files)
    citation_file = None
    aspect_file = None
    for filename in files:
        if "ASPECT" in str(filename):
            assert aspect_file is None
            aspect_file = filename
        if "CITATION_PURPOSE" in str(filename):
            assert citation_file is None
            citation_file = filename
    assert aspect_file is not None and citation_file is not None
    aspect = read_xml(str(aspect_file))
    citation = read_xml(str(citation_file))
    return aspect, citation


def get_sentence_and_tags(aspect: _Element, citation: _Element) -> List[Tuple[str, str]]:
    result = []
    # Abstract processing
    abstract = None
    for child in aspect.getchildren():
        if child.tag == "Abstract":
            abstract = child
            break
    if abstract != None:
        for child in abstract.getchildren():
            if child.tag == "Sentence":
                if child.attrib["aspectClass"] in ["ADVANTAGE", "DISADVANTAGE"]:
                    result.append((child.text, "Evidence"))
                else:
                    result.append((child.text, "Neutral"))

    for achild in aspect.getchildren():
        curtag = "Neutral"
        if achild.tag != "Sentence":
            continue
        # If it is advantage or disadvantage mark it as evidence
        if achild.attrib["aspectClass"] in ["ADVANTAGE", "DISADVANTAGE"]:
            curtag = "Evidence"
        # If it is not advantage or disadvantage check for non-neutral citation
        # We always check to compare number of citations and found citations
        # Cit_context is part of citation
        # Citation is InlineCitation
        for child in citation.getchildren():
            if child.tag == "Cit_context" and child.text in achild.text:
                citation_types = child.attrib.values()
                # We need at least one non NEUTRAL Citation
                allneutral = all(x == "NEUTRAL" for x in citation_types)  # True if all are neutral
                if not allneutral:
                    curtag="Evidence"
                    # curtag = "Citation"
        if curtag is None:
            curtag = "Neutral"
        result.append((achild.text, curtag))
    return result


def main(drinventor_folder: Path):
    dataframe = defaultdict(list)
    for folder in drinventor_folder.glob("A??"):
        print(folder)
        sentences_and_classes: List[Tuple[str, str]] = get_sentence_and_tags(*read_folder(folder))
        for sentence, sentence_class in sentences_and_classes:
            dataframe["topic"].append(None)
            dataframe["sentence"].append(sentence)
            sentence_class = "Neutral" if sentence_class is None else sentence_class
            dataframe["class"].append(sentence_class)
    df = pd.DataFrame(dataframe)
    evidence = (df["class"] != "Neutral").sum()
    print(f"Finished dataset processing, {len(df)} sentences written, {evidence} evidence sentences.")
    output_file = "drinventor.csv"
    df.to_csv(output_file)
    print("Data written to", output_file)


if __name__ == "__main__":
    typer.run(main)
