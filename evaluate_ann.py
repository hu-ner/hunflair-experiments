import argparse
import itertools

from collections import defaultdict
from pathlib import Path
from typing import Callable, List, Tuple, Optional, Dict


TYPE_MAPPING = {
    "chemical": "chemical",
    "disease": "disease",
    "gene": "gene",
    "species": "species"
}

class Metric:
    def __init__(self, name, beta=1):
        self.name = name
        self.beta = beta

        self._tps = defaultdict(int)
        self._fps = defaultdict(int)
        self._tns = defaultdict(int)
        self._fns = defaultdict(int)

    def add_tp(self, class_name):
        self._tps[class_name] += 1

    def add_tn(self, class_name):
        self._tns[class_name] += 1

    def add_fp(self, class_name):
        self._fps[class_name] += 1

    def add_fn(self, class_name):
        self._fns[class_name] += 1

    def get_tp(self, class_name=None):
        if class_name is None:
            return sum([self._tps[class_name] for class_name in self.get_classes()])
        return self._tps[class_name]

    def get_tn(self, class_name=None):
        if class_name is None:
            return sum([self._tns[class_name] for class_name in self.get_classes()])
        return self._tns[class_name]

    def get_fp(self, class_name=None):
        if class_name is None:
            return sum([self._fps[class_name] for class_name in self.get_classes()])
        return self._fps[class_name]

    def get_fn(self, class_name=None):
        if class_name is None:
            return sum([self._fns[class_name] for class_name in self.get_classes()])
        return self._fns[class_name]

    def precision(self, class_name=None):
        if self.get_tp(class_name) + self.get_fp(class_name) > 0:
            return (
                self.get_tp(class_name)
                / (self.get_tp(class_name) + self.get_fp(class_name))
            )
        return 0.0

    def recall(self, class_name=None):
        if self.get_tp(class_name) + self.get_fn(class_name) > 0:
            return (
                self.get_tp(class_name)
                / (self.get_tp(class_name) + self.get_fn(class_name))
            )
        return 0.0

    def f_score(self, class_name=None):
        if self.precision(class_name) + self.recall(class_name) > 0:
            return (
                (1 + self.beta*self.beta)
                * (self.precision(class_name) * self.recall(class_name))
                / (self.precision(class_name) * self.beta*self.beta + self.recall(class_name))
            )
        return 0.0

    def accuracy(self, class_name=None):
        if (
            self.get_tp(class_name) + self.get_fp(class_name) + self.get_fn(class_name) + self.get_tn(class_name)
            > 0
        ):
            return (
                (self.get_tp(class_name) + self.get_tn(class_name))
                / (
                    self.get_tp(class_name)
                    + self.get_fp(class_name)
                    + self.get_fn(class_name)
                    + self.get_tn(class_name)
                )
            )
        return 0.0

    def micro_avg_f_score(self):
        return self.f_score(None)

    def macro_avg_f_score(self):
        class_f_scores = [self.f_score(class_name) for class_name in self.get_classes()]
        if len(class_f_scores) == 0:
            return 0.0
        macro_f_score = sum(class_f_scores) / len(class_f_scores)
        return macro_f_score

    def micro_avg_accuracy(self):
        return self.accuracy(None)

    def macro_avg_accuracy(self):
        class_accuracy = [
            self.accuracy(class_name) for class_name in self.get_classes()
        ]

        if len(class_accuracy) > 0:
            return sum(class_accuracy) / len(class_accuracy)

        return 0.0

    def get_classes(self) -> List:
        all_classes = set(
            itertools.chain(
                *[
                    list(keys)
                    for keys in [
                        self._tps.keys(),
                        self._fps.keys(),
                        self._tns.keys(),
                        self._fns.keys(),
                    ]
                ]
            )
        )
        all_classes = [
            class_name for class_name in all_classes if class_name is not None
        ]
        all_classes.sort()
        return all_classes

    def to_tsv(self):
        return "{}\t{}\t{}\t{}".format(
            self.precision(), self.recall(), self.accuracy(), self.micro_avg_f_score()
        )

    @staticmethod
    def tsv_header(prefix=None):
        if prefix:
            return "{0}_PRECISION\t{0}_RECALL\t{0}_ACCURACY\t{0}_F-SCORE".format(prefix)

        return "PRECISION\tRECALL\tACCURACY\tF-SCORE"

    @staticmethod
    def to_empty_tsv():
        return "\t_\t_\t_\t_"

    def __str__(self):
        all_classes = self.get_classes()
        all_classes = [None] + all_classes
        all_lines = [
            "{0:<10}\ttp: {1} - fp: {2} - fn: {3} - tn: {4} - precision: {5:.4f} - recall: {6:.4f} - accuracy: {7:.4f} - f1-score: {8:.4f}".format(
                self.name if class_name is None else class_name,
                self.get_tp(class_name),
                self.get_fp(class_name),
                self.get_fn(class_name),
                self.get_tn(class_name),
                self.precision(class_name),
                self.recall(class_name),
                self.accuracy(class_name),
                self.f_score(class_name),
            )
            for class_name in all_classes
        ]
        return "\n".join(all_lines)




def print_results(experiment, metric):
    detailed_result = (
        f"{experiment}\nMICRO_AVG: acc {metric.micro_avg_accuracy():.4f} - f1-score {metric.micro_avg_f_score():.4f}"
        f"{experiment}\nMACRO_AVG: acc {metric.macro_avg_accuracy():.4f} - f1-score {metric.macro_avg_f_score():.4f}"
    )
    for class_name in metric.get_classes():
        detailed_result += (
            f"\n{experiment}: {class_name:<10} tp: {metric.get_tp(class_name)} - fp: {metric.get_fp(class_name)} - "
            f"fn: {metric.get_fn(class_name)} - tn: {metric.get_tn(class_name)} - precision: "
            f"{metric.precision(class_name):.4f} - recall: {metric.recall(class_name):.4f} - "
            f"accuracy: {metric.accuracy(class_name):.4f} - f1-score: "
            f"{metric.f_score(class_name):.4f}"
        )

    print(detailed_result)

def copy_dict(dictionary: Dict[str, List[Tuple]]) -> Dict[str, List[Tuple]]:
    copy = {}

    for key, values in dictionary.items():
        value_copy = [v for v in values]
        copy[key] = value_copy

    return copy


def read_annotations(ann_file: Path, add_mention: bool = False,
                     type_mapping: Dict = TYPE_MAPPING) -> Dict[str, List[Tuple]]:
    annotations = defaultdict(list)

    with open(str(ann_file), "r") as ann_reader:
        for line in ann_reader.readlines():
            if not line.strip():
                continue

            columns = line.strip().split("\t")
            document_id, start, end, mention_text, type = columns[:5]
            start, end = int(start), int(end)

            type = type.lower()
            if type not in type_mapping:
                continue

            type = type_mapping[type.lower()]

            if add_mention:
                annotation = (document_id, start, end, type, mention_text)
            else:
                annotation = (document_id, start, end, type)

            annotations[document_id] += [annotation]

    return annotations


def read_corpus(text_file: Path):
    print(f"Checking {text_file}")
    documents = {}
    with open(str(text_file), "r") as txt_reader:
        for line in txt_reader.readlines():
            document_id, text = line.strip().split("\t")
            documents[document_id] = text
    print(f"Found {len(documents)} documents")

    return documents


def check_annotations(text_file: Path, ann_file: Path,
                      type_mapping: Dict = TYPE_MAPPING) -> None:
    documents = read_corpus(text_file)

    print(f"Checking {ann_file}")
    incorrect_spans = 0
    num_annotations = 0

    annotations = read_annotations(ann_file, add_mention=True, type_mapping=type_mapping)
    for _, annotations in annotations.items():
        num_annotations += len(annotations)

        for annotation in annotations:
            document_id, start, end, _ , mention_text = annotation
            if mention_text != documents[document_id][start:end]:
                incorrect_spans += 1

    print(f"Found {incorrect_spans} / {num_annotations} ({(incorrect_spans/num_annotations)*100:.2f}%) incorrect spans\n\n")


def evaluate_files(gold_file: Path, pred_file: Path, match_func: Callable[[Tuple, List], Tuple]) -> Metric:
    gold_annotations = read_annotations(gold_file)
    pred_annotations = read_annotations(pred_file)

    return evaluate(gold_annotations, pred_annotations, match_func)

def evaluate(gold_annotations: Dict[str, List[Tuple]], pred_annotations: Dict[str, List[Tuple]],
             match_func: Callable[[Tuple, List], Tuple]) -> Metric:

    metric = Metric("Evaluation", beta=1)

    copy_gold = copy_dict(gold_annotations)
    for document_id, annotations in pred_annotations.items():
        for pred_entry in annotations:
            # Documents may not contain any gold entity!
            if document_id in copy_gold:
                matched_gold = match_func(pred_entry, copy_gold[document_id])
            else:
                matched_gold = None

            if matched_gold:
                # Assert same document and same entity type!
                assert matched_gold[0] == pred_entry[0] and matched_gold[3] == pred_entry[3]

                copy_gold[document_id].remove(matched_gold)
                metric.add_tp(pred_entry[3])
            else:
                metric.add_fp(pred_entry[3])

    copy_pred = copy_dict(pred_annotations)

    for document_id, annotations in gold_annotations.items():
        for gold_entry in annotations:
            if document_id in copy_pred:
                matched_pred = match_func(gold_entry, copy_pred[document_id])
            else:
                matched_pred = None

            if not matched_pred:
                metric.add_fn(gold_entry[3])
            else:
                # Assert same document and same entity type!
                assert matched_pred[0] == gold_entry[0] and matched_pred[3] == gold_entry[3]

                copy_pred[document_id].remove(matched_pred)

    return metric


def exact_match(entry: Tuple, candidates: List[Tuple]) -> Optional[Tuple]:
    return entry if entry in candidates else None

def partial_match(threshold):
    def _partial_match(entry, candidates):
        for c in candidates:
            if (
                entry[0] == c[0]  # same document?
                and
                entry[3] == c[3]  # same entity type?
                and
                (
                    (abs(entry[1] - c[1]) <= threshold)  # Start offset difference within threshold
                    and
                    (abs(entry[2] - c[2]) <= threshold)  # End offset difference within threshold
                )
            ):
                return c
    return _partial_match


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text_file", required=True)
    parser.add_argument("--gold_file", required=True)
    parser.add_argument("--pred_file", required=True)
    args = parser.parse_args()

    text_file = Path(args.text_file)
    gold_file = Path(args.gold_file)
    pred_file = Path(args.pred_file)

    check_annotations(text_file, gold_file)
    check_annotations(text_file, pred_file)

    # print("Exact matching results:")
    # result = evaluate(gold_file, pred_file, exact_match)
    # print_results("EXACT", result)

    print("Partial matching (t=1) results:")
    result = evaluate_files(gold_file, pred_file, partial_match(1))
    print_results("PARTIAL", result)
