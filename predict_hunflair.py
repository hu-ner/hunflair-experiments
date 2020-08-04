import argparse

from flair.models.sequence_tagger_model import MultiTagger
from flair.tokenization import SciSpacySentenceSplitter
from pathlib import Path
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True, type=Path)
    parser.add_argument("--output_file", required=True, type=Path)
    args = parser.parse_args()

    sentence_splitter = SciSpacySentenceSplitter()
    tagger = MultiTagger.load("hunflair")

    args.output_file.parent.mkdir(parents=True, exist_ok=True)

    with args.input_file.open("r") as f_in, args.output_file.open("w") as f_out:
        lines = f_in.readlines()
        for line in tqdm(lines, total=len(lines)):
            fname, text = line.split("\t")
            sentences = sentence_splitter.split(text)
            tagger.predict(sentences)

            for sentence in sentences:
                for entity in tagger.get_all_spans(sentence):
                    start = entity.start_pos + sentence.start_pos
                    end = entity.end_pos + sentence.start_pos
                    f_out.write(f"{fname}\t{start}\t{end}\t{text[start:end]}\t{entity.tag}\n")
