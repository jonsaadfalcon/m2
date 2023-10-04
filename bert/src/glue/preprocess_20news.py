'''
Script to download and preprocess the 20 News dataset, following the recipe from here: https://github.com/coastalcph/trldc

First, download the dataset "by date":

    mkdir 20news
    cd 20news
    wget http://qwone.com/~jason/20Newsgroups/20news-bydate.tar.gz
    tar -xvf 20news-bydate.tar.gz

    mv 20news-bydate-train train
    mv 20news-bydate-test test

Then, run this script and fill in the INPUT_DIR and the OUTPUT_DIR below.
'''

import os
import random
import json, numpy

INPUT_DIR = '/data/20news'
OUTPUT_DIR = '/data/20news/splits'

examples = []
for label in os.listdir(os.path.join(INPUT_DIR, "train")):
    for filename in os.listdir(os.path.join(INPUT_DIR, "train", label)):
        text = open(os.path.join(INPUT_DIR, "train", label, filename), encoding="latin-1").read()
        examples.append({"text": text.replace("\n", " "), "label": label})
    
print('Train examples:', len(examples))

random.seed(52)
random.shuffle(examples)

dev_size = int(len(examples) * 0.1)
train_set = examples[:-dev_size]
dev_set = examples[-dev_size:]

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

write_list_to_json_file(train_set, os.path.join(OUTPUT_DIR, "train.json"))
write_list_to_json_file(dev_set, os.path.join(OUTPUT_DIR, "dev.json"))

examples = []
for label in os.listdir(os.path.join(INPUT_DIR, "test")):
    for filename in os.listdir(os.path.join(INPUT_DIR, "test", label)):
        text = open(os.path.join(INPUT_DIR, "test", label, filename), encoding="latin-1").read()
        examples.append({"text": text.replace("\n", " "), "label": label})

write_list_to_json_file(examples, os.path.join(OUTPUT_DIR, "test.json"))