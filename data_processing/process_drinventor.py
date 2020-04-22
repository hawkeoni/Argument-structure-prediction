import sys
import os
from typing import List, Tuple

from lxml import etree
from lxml.etree import _Element


def read_xml(filename: str):
    text = open(filename, "rb").read()
    return etree.fromstring(text)


def read_folder(folder: str) -> Tuple[_Element, _Element]:
    files = os.listdir(folder)
    citation_file = None
    aspect_file = None
    for filename in files:
        if "ASPECT" in filename:
            assert aspect_file is None
            aspect_file = filename
        if "CITATION_PURPOSE" in filename:
            assert citation_file is None
            citation_file = filename
    assert aspect_file is not None and citation_file is not None
    aspect = read_xml(os.path.join(folder, aspect_file))
    citation = read_xml(os.path.join(folder, citation_file))
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
                    result.append((child.text, "None"))

    for achild in aspect.getchildren():
        curtag = None
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
                    curtag = "Citation"
        if curtag is None:
            curtag = "None"
        result.append((achild.text, curtag))
    # print(result)
    return result


def main(drinventor_folder: str):
    n = 0
    p = 0
    f = open("drinventor_full.txt", "w", encoding="utf8")
    for i in range(1, 41):
        folder = os.path.join(drinventor_folder, f"A{i:02d}")
        print(folder)
        sentences_and_classes: List[Tuple[str, str]] = get_sentence_and_tags(*read_folder(folder))
        for sentence, sentence_class in sentences_and_classes:
            n += 1
            if sentence_class == "Evidence":
                p += 1
            f.write(f"{sentence}\t{sentence_class}\n")
    print(f"Finished dataset processing, {n} sentences written, {p} evidence sentences.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python process_drinventor.py drinventor_folder")
    else:
        main(sys.argv[1])



