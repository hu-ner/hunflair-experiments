import argparse
import os
import spacy

from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from spacy.language import Language
from tqdm import tqdm

TYPE_MAPPINGS = {
    "en_ner_bionlp13cg_md": {
        "gene_or_gene_product": "gene",
        "organism": "species",
        "cancer": "disease",
        "simple_chemical": "chemical",
        "amino_acid": "chemical"
    },

    "en_ner_bc5cdr_md": {
        "chemical": "chemical",
        "disease": "disease"
    },

    "en_ner_jnlpba_md": {
        "rna" : "gene",
        "dna": "gene",
        "protein": "gene"
    },

    "en_ner_craft_md": {
        "chebi": "chemical",
        "ggp": "gene",
        "taxon": "species",
    }
}


def read_documents(input_file: Path) -> Dict[str, str]:
    documents = {}
    with open(str(input_file), "r") as reader:
        for line in reader.readlines():
            document_id, text = line.strip().split("\t")
            documents[document_id] = text

    return documents


def tag_documents(documents: Dict, model: Language, type_mapping: Dict) -> Dict[str, List]:
    annotations = defaultdict(list)
    for document_id, text in tqdm(documents.items(), total=len(documents)):
        spacy_doc = model(text)

        for entity in spacy_doc.ents:
            entity_type = entity.label_.lower()
            if entity_type not in type_mapping:
                #print(entity_type)
                continue

            annotations[document_id] += [(
                document_id,
                str(entity.start_char),
                str(entity.end_char),
                entity.text,
                type_mapping[entity_type]
            )]

    return annotations


def write_annotations(annotations: Dict[str, List], output_file: Path):
    with open(str(output_file), "w") as writer:
        for _, annotations in annotations.items():
            for annotation in annotations:
                writer.write("\t".join(list(annotation)) + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output_file", required=True)
    args = parser.parse_args()

    spacy_model = spacy.load(args.model)
    type_mapping = TYPE_MAPPINGS[args.model]

    documents = read_documents(Path(args.input_file))
    annotations = tag_documents(documents, spacy_model, type_mapping)

    output_file = Path(args.output_file)
    os.makedirs(str(output_file.parent), exist_ok=True)

    write_annotations(annotations, output_file)

