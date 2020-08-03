import flair
import os

from flair.datasets import BIONLP2013_CG, BIONLP2013_PC, PDR
from lxml import etree
from pathlib import Path
from typing import List, Dict


def prepare_craft_corpus(output_dir: Path):
    craft_dir = Path(os.path.expanduser(flair.cache_root)) / "datasets" / "craft_v4" / "CRAFT-4.0.1"

    text_dir = craft_dir / "articles" / "txt"
    documents = {}
    with open(os.path.join(str(output_dir), "craft_v4.tsv"), "w") as writer:
        for file in text_dir.iterdir():
            if not str(file).endswith(".txt"):
                continue

            document_id = file.stem
            file_text = open(str(file)).read().replace("\n", " ").replace("\t", " ").strip()

            writer.write(f"{document_id}\t{file_text}\n")
            if document_id in documents:
                raise AssertionError()

            documents[document_id] = file_text

    annotation_dir = craft_dir / "concept-annotation"
    annotation_dirs = {
        ("Gene", annotation_dir / "PR" / "PR"),
        ("Chemical", annotation_dir / "CHEBI" / "CHEBI"),
        ("Species", annotation_dir / "NCBITaxon" / "NCBITaxon")
    }

    ann_writer = open(os.path.join(str(output_dir), f"craft_v4.ann"), "w")
    for type, type_dir in annotation_dirs:
        for file in (type_dir / "knowtator").iterdir():
            document_id = str(file.name).split(".")[0]

            root = etree.parse(str(file))
            for annotation in root.xpath(".//annotation"):
                text_element = annotation.find("spannedText")
                mention_texts = text_element.text.split(" ... ")

                for i, span_element in enumerate(annotation.findall("span")):
                    start = int(span_element.get("start"))
                    end = int(span_element.get("end"))
                    span_text = mention_texts[i]

                    ann_writer.write(f"{document_id}\t{start}\t{end}\t{span_text}\t{type}\n")

                    if span_text != documents[document_id][start:end]:
                        print(span_text)
                        print(documents[document_id][start:end])
                        print()

                    assert span_text == documents[document_id][start:end]

    ann_writer.close()


def prepare_brat_corpus(corpus_name: str,
                        input_dirs: List[Path],
                        ann_file_suffixes: List[str],
                        entity_type_mapping: Dict[str, str],
                        output_dir: Path):
    documents = {}
    annotations = set()

    for input_dir in input_dirs:
        text_files = list(input_dir.glob("*.txt"))

        for text_file in text_files:
            document_text = open(str(text_file)).read().replace("\n", " ").replace("\t", " ").strip()
            document_id = text_file.stem

            if document_id in documents:
                raise AssertionError()

            documents[document_id] = document_text

            for suffix in ann_file_suffixes:
                ann_file = text_file.with_suffix(suffix)
                if not ann_file.is_file():
                    continue

                with open(str(ann_file), "r") as ann_file:
                    for line in ann_file:
                        fields = line.strip().split("\t")

                        # Ignore empty lines or relation annotations
                        if not fields or len(fields) <= 2:
                            continue

                        mention_text = fields[2].strip()

                        ent_type, char_start, char_end = fields[1].split()
                        if ent_type not in entity_type_mapping:
                            continue

                        start = int(char_start)
                        end = int(char_end)

                        # FIX annotation of whitespaces (necessary for PDR)
                        while document_text[start:end].startswith(" "):
                            start += 1

                        while document_text[start:end].endswith(" "):
                            end -= 1

                        assert document_text[start:end] == mention_text
                        annotations.add((document_id, str(start), str(end), mention_text, entity_type_mapping[ent_type]))

    writer = open(str(output_dir / f"{corpus_name}.tsv"), "w")
    for document_id, text in documents.items():
        writer.write(f"{document_id}\t{text}\n")
    writer.close()

    writer = open(str(output_dir / f"{corpus_name}.ann"), "w")
    annotations = sorted(annotations)
    for annotation in annotations:
        writer.write("\t".join(annotation) + "\n")
    writer.close()


def prepare_pdr_corpus(output_dir: Path):
    pdr_dir = Path(os.path.expanduser(flair.cache_root)) / "datasets" / "pdr" / "Plant-Disease_Corpus"

    prepare_brat_corpus(
        corpus_name="pdr",
        input_dirs=[pdr_dir],
        ann_file_suffixes=[".ann", ".ann2"],
        entity_type_mapping={"Disease": "Disease"},
        output_dir=output_dir
    )


def prepare_bionlp13cg_corpus(output_dir: Path):
    bionlp213cg_dir = Path(os.path.expanduser(flair.cache_root)) / "datasets" / "bionlp2013_cg" / "original"

    train_dir = bionlp213cg_dir / "BioNLP-ST_2013_CG_training_data"
    dev_dir = bionlp213cg_dir / "BioNLP-ST_2013_CG_development_data"
    test_dir = bionlp213cg_dir / "BioNLP-ST_2013_CG_test_data"

    prepare_brat_corpus(
        corpus_name="bionlp2013cg",
        input_dirs=[train_dir, dev_dir, test_dir],
        ann_file_suffixes=[".a1", ".a2"],
        entity_type_mapping={
            "Gene_or_gene_product": "Gene",
            "Organism": "Species",
            "Cancer": "Disease",
            "Simple_chemical": "Chemical",
            "Amino_acid": "Chemical"
        },
        output_dir=output_dir
    )


if __name__ == "__main__":
    output_directory = Path("./corpora")
    os.makedirs(output_directory, exist_ok=True)

    # Download the data sets via Flair
    PDR()
    BIONLP2013_CG()
    BIONLP2013_PC()

    # Prepare the gold standard annotations
    prepare_craft_corpus(output_directory)
    prepare_pdr_corpus(output_directory)
    prepare_bionlp13cg_corpus(output_directory)
