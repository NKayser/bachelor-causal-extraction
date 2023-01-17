import json

import typer
from pathlib import Path
from tqdm import tqdm

from spacy.tokens import Span, DocBin, Doc
from spacy.vocab import Vocab
from wasabi import Printer
#from spacy.tokenizer import Tokenizer
from spacy.lang.en import English
from spacy.util import compile_infix_regex
import re
import spacy

nlp = spacy.blank("en")
nlp.add_pipe("sentencizer")
# Create a blank Tokenizer with just the English vocab

msg = Printer()

ann = "assets/labels_and_predictions.jsonl"
train_file = 'data/my_train.spacy'
dev_file = 'data/my_dev.spacy'
test_file = 'data/my_test.spacy'


def ent_is_in_sent(ent, sent):
    return not (ent.end_char < sent.start_char or ent.start_char > sent.end_char
                or ent.start_char < sent.start_char or ent.end_char > sent.end_char)


def get_ents_of_sent(all_entities, sent):
    return [ent for ent in all_entities if ent_is_in_sent(ent, sent)]


def get_ents_of_relation(all_ent_obj, relation):
    from_id = relation["from_id"]
    to_id = relation["to_id"]
    from_ent = None
    to_ent = None

    for ent in all_ent_obj:
        if from_id == ent["id"]:
            from_ent = ent
        if to_id == ent["id"]:
            to_ent = ent
        if from_ent is not None and to_ent is not None:
            break
    assert from_ent is not None and to_ent is not None
    return from_ent, to_ent


def main(json_loc: Path, train_file: Path, dev_file: Path, test_file: Path):
    Doc.set_extension("rel", default={}, force=True)

    docs = {"train": [], "dev": [], "test": [], "total": []}
    ids = {"train": set(), "dev": set(), "test": set(), "total": set()}
    count_all = {"train": 0, "dev": 0, "test": 0, "total": 0}
    count_pos = {"train": 0, "dev": 0, "test": 0, "total": 0}

    with open(json_loc, encoding="utf8") as jsonfile:
        json_list = list(jsonfile)
        for line in tqdm(json_list):
            article_obj = json.loads(line)
            article_id = article_obj["id"]
            relations = list(filter(lambda x: x["type"] == "causality", article_obj["relations"]))
            entities = article_obj["labeled_entities"]
            article_doc = nlp(article_obj["text"])
            article_doc.spans["sc"] = [article_doc.char_span(x["start_offset"], x["end_offset"], x["label"],
                                                             alignment_mode="expand")
                                       for x in entities]

            # each sentence will become its own document. Relations across sentences cannot be captured for now.
            for sent in article_doc.sents:
                span_starts = set()
                # Parse the tokens
                tokens = nlp(sent.text)

                spaces = [True if tok.whitespace_ else False for tok in tokens]
                words = [t.text for t in tokens]
                doc = Doc(nlp.vocab, words=words, spaces=spaces)

                old_cause_effect_spans = list(filter(lambda x: x.label_ in ["cause", "effect"],
                                                     get_ents_of_sent(article_doc.spans["sc"], sent)))

                cause_effect_spans = [doc.char_span(x.start_char - sent.start_char,
                                                    x.end_char - sent.start_char,
                                                    x.label_, alignment_mode="expand")
                                      for x in old_cause_effect_spans]
                for ent in cause_effect_spans:
                    if ent is None:
                        print("ent is None")
                        print(sent)
                        print(old_cause_effect_spans)
                        assert False

                if len(cause_effect_spans) > 0:
                    print(cause_effect_spans)
                if len(cause_effect_spans) == 0:
                    count_all["train"] += 1
                    docs["train"].append(doc)

                for span in cause_effect_spans:
                    span_starts.add(span[0].i)

                previous_end = 0
                overlap = False
                for ent in cause_effect_spans:
                    if ent.start_char <= previous_end:
                        overlap = True
                        break
                    previous_end = ent.end_char
                if overlap:
                    continue
                doc.ents = cause_effect_spans

                # Parse the relations
                rels = {}
                for x1 in span_starts:
                    for x2 in span_starts:
                        rels[(x1, x2)] = {}
                        # print(rels)
                for relation in relations:
                    from_ent, to_ent = get_ents_of_relation(entities, relation)

                    if not (from_ent["label"] == "cause" and to_ent["label"] == "effect"):
                        # cannot deal with "core reference" yet
                        continue

                    from_ent = article_doc.char_span(from_ent["start_offset"],
                                                     from_ent["end_offset"],
                                                     from_ent["label"], alignment_mode="expand")
                    to_ent = article_doc.char_span(to_ent["start_offset"],
                                                   to_ent["end_offset"],
                                                   to_ent["label"], alignment_mode="expand")
                    from_ent_sent = doc.char_span(from_ent.start_char - sent.start_char,
                                                  from_ent.end_char - sent.start_char,
                                                  from_ent.label_, alignment_mode="expand")
                    to_ent_sent = doc.char_span(to_ent.start_char - sent.start_char,
                                                to_ent.end_char - sent.start_char,
                                                to_ent.label_, alignment_mode="expand")

                    # cannot deal with realtion across sentences yet
                    if not (ent_is_in_sent(from_ent, sent) and ent_is_in_sent(to_ent, sent)):
                        continue

                    start = from_ent_sent[0].i
                    end = to_ent_sent[0].i
                    # print(rels[(start, end)])
                    # print(label)
                    if "causality" not in rels[(start, end)]:
                        rels[(start, end)]["causality"] = 1.0
                        count_pos["train"] += 1
                        # print(pos)
                        # print(rels[(start, end)])

                # The annotation is complete, so fill in zero's where the data is missing
                for x1 in span_starts:
                    for x2 in span_starts:
                        if "causality" not in rels[(x1, x2)]:
                            rels[(x1, x2)]["causality"] = 0.0

                doc._.rel = rels
                if len(doc._.rel.items()) > 0:
                    print(doc._.rel)

                docs["train"].append(doc)
                count_all["train"] += 1

    # print(len(docs["total"]))
    docbin = DocBin(docs=docs["train"], store_user_data=True)
    docbin.to_disk(train_file)
    msg.info(
        f"{len(docs['train'])} training sentences, {count_pos['train']}/{count_all['train']} positive"
    )


main(ann, train_file, dev_file, test_file)