from pathlib import Path
from tqdm import tqdm
import pandas as pd

from spacy.tokens import DocBin, Doc

import spacy
from wasabi import Printer


msg = Printer()


nlp = spacy.blank("en")
# Create a blank Tokenizer with just the English vocab


crest_xls = 'assets/crest_v2.xlsx'
train_file = 'data/crest_train.spacy'
dev_file = 'data/crest_dev.spacy'
test_file = 'data/crest_test.spacy'


def span_info_to_list(str):
    span_info = str.split()[1:]
    span_info = [[int(y) for y in x.split(":")] for x in span_info]
    return span_info


def ent_to_token_slice(doc, ent):
    span = doc.char_span(ent.start_char, ent.end_char, alignment_mode="expand")
    return span[0].i, span[-1].i + 1


def main(excel_loc: str, train_file: str, dev_file: str, test_file: str):
    Doc.set_extension("rel", default={})

    docs = {"train": [], "dev": [], "test": []}
    ids = {"train": set(), "dev": set(), "test": set(), "skipped": set()}
    count_all = {"train": 0, "dev": 0, "test": 0}
    count_pos = {"train": 0, "dev": 0, "test": 0}

    crest_df = pd.read_excel(excel_loc, engine="openpyxl")
    crest_df = crest_df.reset_index()
    for index, row in tqdm(crest_df.iterrows()):
        context = row["context"]
        direction = row["direction"]
        label = row["label"]
        split = row["split"]
        global_id = row["global_id"]
        span1_info, span2_info, signal_info = row["idx"].splitlines()
        span1_info = span_info_to_list(span1_info)
        span2_info = span_info_to_list(span2_info)
        # skip docs with multiple cause/effect spans and missing spans
        if len(span1_info) != 1 or len(span2_info) != 1 or span1_info[0][0] == -1 or span2_info[0][0] == -1 or direction == -1:
            ids["skipped"].add(global_id)
            continue
        span1_info = span1_info[0]
        span2_info = span2_info[0]

        span_starts = set()
        neg = 0
        pos = 0
        # Parse the tokens
        tokens = nlp(context)

        spaces = [True if tok.whitespace_ else False for tok in tokens]
        words = [t.text for t in tokens]
        doc = Doc(nlp.vocab, words=words, spaces=spaces)

        entities = []
        for span in [span1_info, span2_info]:
            entity = doc.char_span(span[0], span[1], label="event", alignment_mode="expand")
            #token_slice = ent_to_token_slice(tokens, entity)
            # print(span_end_to_start)
            entities.append(entity)
            span_starts.add(entity[0].i)

        if entities[0].end_char >= entities[1].start_char or abs(list(span_starts)[0] - list(span_starts)[1]) > 100:
            ids["skipped"].add(global_id)
            continue
        doc.ents = entities

        # Parse the relations
        rels = {}
        for x1 in span_starts:
            for x2 in span_starts:
                rels[(x1, x2)] = {}
                if label == 1 and ((x1 < x2 and direction >= 0) or (x1 > x2 and direction <= 0)):
                    rels[(x1, x2)]["causes"] = 1.0
                else:
                    rels[(x1, x2)]["causes"] = 0.0
        doc._.rel = rels
        #print(doc._.rel)

        if label == 1:
            pos += 1
        else:
            neg += 1

        if split == 1:
            count_pos["dev"] += pos
            count_all["dev"] += pos + neg
            docs["dev"].append(doc)
            ids["dev"].add(global_id)
        elif split == 2:
            count_pos["test"] += pos
            count_all["test"] += pos + neg
            docs["test"].append(doc)
            ids["test"].add(global_id)
        else: # split == 0 or null
            count_pos["train"] += pos
            count_all["train"] += pos + neg
            docs["train"].append(doc)
            ids["train"].add(global_id)


    docbin = DocBin(docs=docs["train"], store_user_data=True)
    docbin.to_disk(train_file)
    msg.info(
        f"{len(docs['train'])} training sentences from {len(ids['train'])} articles, "
        f"{count_pos['train']}/{count_all['train']} pos instances."
    )

    docbin = DocBin(docs=docs["dev"], store_user_data=True)
    docbin.to_disk(dev_file)
    msg.info(
        f"{len(docs['dev'])} dev sentences from {len(ids['dev'])} articles, "
        f"{count_pos['dev']}/{count_all['dev']} pos instances."
    )

    docbin = DocBin(docs=docs["test"], store_user_data=True)
    docbin.to_disk(test_file)
    msg.info(
        f"{len(docs['test'])} test sentences from {len(ids['test'])} articles, "
        f"{count_pos['test']}/{count_all['test']} pos instances."
    )

    msg.info(
        f"Skipped {len(ids['skipped'])} sentences"
    )


main(crest_xls, train_file, dev_file, test_file)
