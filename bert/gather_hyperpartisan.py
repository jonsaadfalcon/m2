
import os
import xml.etree.ElementTree as ET

INPUT_DIR = "datasets/hyperpartisan"
OUTPUT_DIR = "datasets/hyperpartisan"

articles = ET.parse(os.path.join(INPUT_DIR, "articles-training-byarticle-20181122.xml")).getroot().findall("article")
labels = ET.parse(os.path.join(INPUT_DIR, "ground-truth-training-byarticle-20181122.xml")).getroot().findall("article")
assert len(articles) == len(labels)



import re

FLAGS = re.MULTILINE | re.DOTALL

def re_sub(pattern, repl, text, flags=None):
    if flags is None:
        return re.sub(pattern, repl, text, flags=FLAGS)
    else:
        return re.sub(pattern, repl, text, flags=(FLAGS | flags))

def clean_txt(text):

    text = re.sub(r"[a-zA-Z]+\/[a-zA-Z]+", " ", text)
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"&#160;", "", text)

    # Remove URL
    text = re_sub(r"(http)\S+", "", text)
    text = re_sub(r"(www)\S+", "", text)
    text = re_sub(r"(href)\S+", "", text)
    # Remove multiple spaces
    text = re_sub(r"[ \s\t\n]+", " ", text)

    # remove repetition
    text = re_sub(r"([!?.]){2,}", r"\1", text)
    text = re_sub(r"\b(\S*?)(.)\2{2,}\b", r"\1\2", text)

    return text.strip()




from tqdm import tqdm

data = {}
for article, label in tqdm(zip(articles, labels), total=len(articles), desc="preprocessing"):
    text = ET.tostring(article, method="text", encoding="utf-8").decode("utf-8")
    text = clean_txt(text)
    id_ = int(label.attrib["id"])
    data[id_] = {"text": text, "label": label.attrib["hyperpartisan"], "id": id_}



import json
from collections import defaultdict

splits = defaultdict(list)
for s, ids in json.load(open(os.path.join(INPUT_DIR, "hp-splits.json"))).items():
    for i in ids:
        splits[s].append(data[i])



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

os.makedirs(OUTPUT_DIR, exist_ok=True)

for s, data in splits.items():
    write_list_to_json_file(data, os.path.join(OUTPUT_DIR, f"{s}.json"))