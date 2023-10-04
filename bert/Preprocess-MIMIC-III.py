
INPUT_DIR = "datasets/mimiciii/"
OUTPUT_DIR = "datasets/mimiciii/0"
SPLIT_DIR = "datasets/mimiciii/split"

import csv, operator, os, re
from collections import Counter
from nltk.tokenize import RegexpTokenizer
from tqdm import tqdm
import pandas as pd

def reformat_code(code, is_diagnosis):
    """Generally, procedure codes have dots after the first two digits,
    while diagnosis codes have dots after the first three digits."""
    code = "".join(code.split("."))
    if is_diagnosis:
        if code.startswith("E"):
            if len(code) > 4:
                code = code[:4] + "." + code[4:]
        else:
            if len(code) > 3:
                code = code[:3] + "." + code[3:]
    else:
        code = code[:2] + "." + code[2:]
    return code

PROCEDURES_ICD = pd.read_csv(os.path.join(INPUT_DIR, "PROCEDURES_ICD.csv"))
DIAGNOSES_ICD = pd.read_csv(os.path.join(INPUT_DIR, "DIAGNOSES_ICD.csv"))
DIAGNOSES_ICD["absolute_code"] = DIAGNOSES_ICD.apply(lambda row: str(reformat_code(str(row[4]), True)), axis=1)
PROCEDURES_ICD["absolute_code"] = PROCEDURES_ICD.apply(lambda row: str(reformat_code(str(row[4]), False)), axis=1)
ALL_ICD = pd.concat([DIAGNOSES_ICD, PROCEDURES_ICD])
ALL_ICD.to_csv("ALL_ICD.csv", index=False,
               columns=["ROW_ID", "SUBJECT_ID", "HADM_ID", "SEQ_NUM", "absolute_code"],
               header=["ROW_ID", "SUBJECT_ID", "HADM_ID", "SEQ_NUM", "ICD9_CODE"])
ALL_ICD = pd.read_csv("ALL_ICD.csv", dtype={"ICD9_CODE": str})
#len(ALL_ICD["ICD9_CODE"].unique())

tokenizer = RegexpTokenizer(r"\w+")
with open(os.path.join(INPUT_DIR, "NOTEEVENTS.csv"), "r") as in_f:
    with open("DISCHARGE_SUMMARIES.csv", "w") as out_f:
        out_f.write(",".join(["SUBJECT_ID", "HADM_ID", "CHARTTIME", "TEXT"]) + "\n")
        reader = csv.reader(in_f)
        next(reader) # skip the first line

        for line in tqdm(reader):
            if line[6] == "Discharge summary":
                text = line[10].strip()
                # tokenize, lowercase and normalize numerics
                text = re.sub("\d", "0", text.lower())
                tokens = tokenizer.tokenize(text)
                # Mullenbach et al delete numeric-only tokens
                text = '"' + ' '.join(tokens) + '"'
                out_f.write(",".join([line[1], line[2], line[4], text]) + "\n")

DISCHARGE_SUMMARIES = pd.read_csv("DISCHARGE_SUMMARIES.csv")
len(DISCHARGE_SUMMARIES["HADM_ID"].unique())

DISCHARGE_SUMMARIES = DISCHARGE_SUMMARIES.sort_values(["SUBJECT_ID", "HADM_ID"])
DISCHARGE_SUMMARIES.to_csv("DISCHARGE_SUMMARIES_SORTED.csv", index=False)
#! rm DISCHARGE_SUMMARIES.csv
ALL_ICD = pd.read_csv("ALL_ICD.csv")
ALL_ICD = ALL_ICD.sort_values(["SUBJECT_ID", "HADM_ID"])
print(str(len(DISCHARGE_SUMMARIES["HADM_ID"].unique())) + "_" + str(len(ALL_ICD["HADM_ID"].unique())))

hadm_ids = set(DISCHARGE_SUMMARIES["HADM_ID"])

with open("ALL_ICD.csv", "r") as in_f:
    with open("ALL_ICD_FILTERED.csv", "w") as out_f:
        writer = csv.writer(out_f)
        writer.writerow(["SUBJECT_ID", "HADM_ID", "ICD9_CODE", "ADMITTIME", "DISCHTIME"])
        reader = csv.reader(in_f)
        next(reader)
        for row in reader:
            hadm_id = int(row[2])
            if hadm_id in hadm_ids:
                writer.writerow(row[1:3] + [row[-1], "", ""])
ALL_ICD_FILTERED = pd.read_csv("ALL_ICD_FILTERED.csv", index_col=None)
len(ALL_ICD_FILTERED["HADM_ID"].unique())

ALL_ICD_FILTERED = ALL_ICD_FILTERED.sort_values(["SUBJECT_ID", "HADM_ID"])
ALL_ICD_FILTERED.to_csv("ALL_ICD_FILTERED_SORTED.csv", index=False)
#! rm ALL_ICD.csv ALL_ICD_FILTERED.csv

def next_labels(label_filepath):
    reader = csv.reader(label_filepath)
    next(reader)

    first_line = next(reader)

    cur_subj = int(first_line[0])
    cur_hadm = int(first_line[1])
    cur_labels = [first_line[2]]

    for row in reader:
        subj_id = int(row[0])
        hadm_id = int(row[1])
        label = row[2]
        # keep reading until you hit a new hadm id
        if hadm_id != cur_hadm or subj_id != cur_subj:
            yield cur_subj, cur_hadm, cur_labels
            cur_subj = subj_id
            cur_hadm = hadm_id
            cur_labels = [label]
        else:
            # add to the labels and move on
            cur_labels.append(label)
    yield cur_subj, cur_hadm, cur_labels


def next_notes(note_filepath):
    reader = csv.reader(note_filepath)
    next(reader)

    first_line = next(reader)

    cur_subj = int(first_line[0])
    cur_hadm = int(first_line[1])
    cur_text = first_line[3]

    for row in reader:
        subj_id = int(row[0])
        hadm_id = int(row[1])
        text = row[3]
        # keep reading until you hit a new hadm id
        if hadm_id != cur_hadm or subj_id != cur_subj:
            yield cur_subj, cur_hadm, cur_text
            cur_subj = subj_id
            cur_hadm = hadm_id
            cur_text = text
        else:
            # concatenate to the discharge summary and move on
            cur_text += " " + text
    yield cur_subj, cur_hadm, cur_text

def concat_data(note_filepath, label_filepath, out_filepath):
    with open(label_filepath, "r") as label_f:
        with open(note_filepath, "r") as note_f:
            with open(out_filepath, "w") as out_f:
                writer = csv.writer(out_f)
                writer.writerow(["SUBJECT_ID", "HADM_ID", "TEXT", "LABELS"])

                labels_gen = next_labels(label_f)
                notes_gen = next_notes(note_f)

                for i, (subj_id, hadm_id, text) in enumerate(notes_gen):
                    cur_subj, cur_hadm , cur_labels= next(labels_gen)

                    assert cur_hadm == hadm_id
                    writer.writerow([subj_id, str(hadm_id), text, ";".join(cur_labels)])

concat_data(note_filepath="DISCHARGE_SUMMARIES_SORTED.csv",
            label_filepath="ALL_ICD_FILTERED_SORTED.csv",
            out_filepath="DISCHARGE_SUMMARIES_ICD.csv")

#! rm DISCHARGE_SUMMARIES_SORTED.csv ALL_ICD_FILTERED_SORTED.csv

DISCHARGE_SUMMARIES_ICD = pd.read_csv("DISCHARGE_SUMMARIES_ICD.csv")
all_tokens = set()
num_tokens = 0
for row in DISCHARGE_SUMMARIES_ICD.itertuples():
    for t in row[3].split():
        all_tokens.add(t)
        num_tokens += 1

#len(all_tokens), num_tokens, len(DISCHARGE_SUMMARIES_ICD["HADM_ID"].unique())

## Create train/dev/test splits
split_ids = {}
for split in ["train", "dev", "test"]:
    lines = [l.strip() for l in open(os.path.join(SPLIT_DIR, f"{split}_full_hadm_ids.csv")).readlines()]
    split_ids[split] = set(lines)
    print(f"{split} set has {len(split_ids[split])} examples")

split_examples = {k: [] for k in split_ids}

with open("DISCHARGE_SUMMARIES_ICD.csv", "r") as f:
    reader = csv.reader(f)
    next(reader)

    for row in reader:
        hadm_id = row[1]
        text = row[2]
        labels = [l.strip() for l in row[3].split(";") if len(l.strip()) > 0]
        labels = list(set(labels))
        if len(labels) == 0:
            print(f"Ignore one record ({hadm_id}), because it has no labels")
            continue
        example = {"subject_id": int(row[0]), "hadm_id": hadm_id, "text": text ,"labels": labels}
        if hadm_id in split_ids["train"]:
            split_examples["train"].append(example)
        elif hadm_id in split_ids["dev"]:
            split_examples["dev"].append(example)
        else:
            assert hadm_id in split_ids["test"]
            split_examples["test"].append(example)

import json, numpy

class NumpyJsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.integer):
            return int(obj)
        elif isinstance(obj, numpy.floating):
            return float(obj)
        elif isinstance(obj, numpy.ndarray):
            return obj.tolist()
        else:
            return super(NumpyJsonEncoder, self).default(obj)

def write_list_to_json_file(data, filepath):
    with open(filepath, "w") as f:
        for i in data:
            f.write(f"{json.dumps(i, cls=NumpyJsonEncoder)}\n")

os.makedirs(os.path.join(OUTPUT_DIR, "full"), exist_ok=True)

for k, v in split_examples.items():
    sorted_v = sorted(v, key=lambda i: len(i["text"].split()))
    write_list_to_json_file(sorted_v, os.path.join(OUTPUT_DIR, "full", f"{k}.json"))

counts = Counter()
DISCHARGE_SUMMARIES_ICD = pd.read_csv("DISCHARGE_SUMMARIES_ICD.csv")
for row in DISCHARGE_SUMMARIES_ICD.itertuples():
    for label in str(row[4]).split(";"):
        counts[label] += 1

import json

os.makedirs(os.path.join(OUTPUT_DIR, "50"), exist_ok=True)
counts_sorted = sorted(counts.items(), key=operator.itemgetter(1), reverse=True)
label2idx = {code[0]: i for i, code in enumerate(counts_sorted)}
json.dump(label2idx, open(os.path.join(OUTPUT_DIR, "full", "label2idx.json"), "w"))
top50labels = [code[0] for code in counts_sorted[:50]]
label2idx = {l: i for i, l in enumerate(top50labels)}
json.dump(label2idx, open(os.path.join(OUTPUT_DIR, "50", "label2idx.json"), "w"))

split_ids = {}
for split in ["train", "dev", "test"]:
    lines = [l.strip() for l in open(os.path.join(SPLIT_DIR, f"{split}_50_hadm_ids.csv")).readlines()]
    split_ids[split] = set(lines)
    print(f"{split} set has {len(split_ids[split])} examples")

split_examples = {k: [] for k in split_ids}

with open("DISCHARGE_SUMMARIES_ICD.csv", "r") as f:
    reader = csv.reader(f)
    next(reader)

    for row in reader:
        hadm_id = row[1]
        text = row[2]
        labels = set(row[3].split(";")).intersection(set(top50labels))
        example = {"subject_id": int(row[0]), "hadm_id": hadm_id, "text": text ,"labels": list(labels)}
        if hadm_id in split_ids["train"]:
            split_examples["train"].append(example)
        elif hadm_id in split_ids["dev"]:
            split_examples["dev"].append(example)
        elif hadm_id in split_ids["test"]:
            split_examples["test"].append(example)
for k, v in split_examples.items():
    sorted_v = sorted(v, key=lambda i: len(i["text"].split()))
    write_list_to_json_file(sorted_v, os.path.join(OUTPUT_DIR, "50", f"{k}.json"))
#!rm DISCHARGE_SUMMARIES_ICD.csv