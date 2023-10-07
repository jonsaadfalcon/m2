
import os
import json
import datasets
from datasets import load_dataset
import transformers
import torch

_task_column_names = {
    '20news': ('text', None),
    'contract_nli': ('input', None),
    'mimic': ('text', None),
    'ecthr': ('text', None),
}

def create_news20_dataset(split):

    OUTPUT_DIR = "datasets/20news/splits/"
    
    print(f"Split: {split}")

    if split == 'train':
        path = os.path.join(OUTPUT_DIR, 'train.json')
    elif 'val' in split or 'dev' in split:
        path = os.path.join(OUTPUT_DIR, 'dev.json')
        split = 'dev'
    elif split == 'test':
        path = os.path.join(OUTPUT_DIR, 'test.json')
    
    dataset = load_dataset(
        'json', 
        data_files=[path]
    )
    dataset = dataset['train']

    label_names = set(dataset['label'])
    label_names = sorted(label_names)
    label_map = {label: i for i, label in enumerate(label_names)}

    print("label_map for news20")
    print(label_map)

    def map_labels(example):
        example['label'] = label_map[example['label']]
        return example
    
    dataset = dataset.map(map_labels)

    return dataset


def create_mimic_dataset(split):
    OUTPUT_DIR = "datasets/mimiciii/0/50"
    
    print(f"Split: {split}")

    if split == 'train':
        path = os.path.join(OUTPUT_DIR, 'train.json')
    elif 'val' in split or 'dev' in split:
        path = os.path.join(OUTPUT_DIR, 'dev.json')
        split = 'dev'
    elif split == 'test':
        path = os.path.join(OUTPUT_DIR, 'test.json')
    
    dataset = load_dataset(
        'json', 
        data_files=[path]
    )
    dataset = dataset['train']

    label2idx_path = os.path.join(OUTPUT_DIR, 'label2idx.json')
    with open(label2idx_path, 'r') as f:
        label_map = json.load(f)

    num_labels = len(label_map)

    def map_labels(example):
        labels = [0 for i in range(num_labels)]
        for label in example['labels']:
            labels[label_map[label]] = 1
        example['label_ids'] = torch.tensor(labels, dtype=torch.long)
        #example['label'] = torch.tensor(labels, dtype=torch.long)
        del example['labels']
        return example
    
    dataset = dataset.map(map_labels)

    #print("Created Mimic Dataset!")
    #print(dataset)
    #print(dataset[0]['label_ids'])
    #print(dataset[0]['text'])

    return dataset

def create_ecthr_dataset(split):
    OUTPUT_DIR = "datasets/ecthr/"
    
    print(f"Split: {split}")

    if split == 'train':
        path = os.path.join(OUTPUT_DIR, 'train.json')
    elif 'val' in split or 'dev' in split:
        path = os.path.join(OUTPUT_DIR, 'dev.json')
        split = 'dev'
    elif split == 'test':
        path = os.path.join(OUTPUT_DIR, 'test.json')
    
    dataset = load_dataset(
        'json', 
        data_files=[path]
    )
    dataset = dataset['train']

    dataset = dataset.rename_column('labels', 'label')

    label_names = set()
    for row in range(len(dataset)):
        assert type(dataset[row]['label']) == list
        #assert len(dataset[row]['label']) == 1
        for label in dataset[row]['label']:
            label_names.add(label)

    #label_names = set(dataset['label'])
    label_names = sorted(label_names)
    label_map = {label: i for i, label in enumerate(label_names)}

    print("label_map for ecthr")
    print(label_map)
    num_labels = len(label_map)

    def map_labels(example):
        labels = [0 for i in range(num_labels)]
        for label in example['label']:
            labels[label_map[label]] = 1
        example['label_ids'] = torch.tensor(labels, dtype=torch.long)
        #example['labels'] = torch.tensor(labels, dtype=torch.long)
        del example['label']
        return example
    
    dataset = dataset.map(map_labels)

    return dataset

def create_contract_nli_dataset(split, max_retries=10):
    download_config = datasets.DownloadConfig(max_retries=max_retries)
    dataset = datasets.load_dataset(
        "tau/scrolls", "contract_nli",
        split=split,
        download_config=download_config,
    )

    # remap 'id' to 'idx'
    dataset = dataset.rename_column('id', 'idx')

    # remap the labels 
    mapping = {
        'Not mentioned': 0, 
        'Entailment': 1, 
        'Contradiction': 2
    }

    def map_labels(example):
        labels = [0 for i in range(len(mapping))]
        labels[mapping[example['output']]] = 1
        example['label_ids'] = torch.tensor(labels, dtype=torch.long)
        #example['label'] = torch.tensor(labels, dtype=torch.long)
        #example['label'] = torch.reshape(example['label'], (3, 1))
        del example['output']
        return example
    
    dataset = dataset.map(map_labels)

    print("Created Contract NLI Dataset!")
    print(dataset)
    #print(dataset[0]['label_ids'])
    #print(dataset[0]['input'])

    return dataset
    
    """ def map_labels(example):
        example['output'] = mapping[example['output']]
        example['output'] = torch.tensor(example['output'], dtype=torch.long)
        return example
    dataset = dataset.map(map_labels)
    dataset = dataset.rename_column('output', 'label')

    print("Contract NLI Dataset")
    print(dataset)

    return dataset """


def create_long_context_dataset(task_name, split, tokenizer_name, max_seq_length, num_workers=8):
    
    if task_name == '20news':
        dataset = create_news20_dataset(split)
    elif task_name == 'mimic':
        dataset = create_mimic_dataset(split)
    elif task_name == 'contract_nli':
        dataset = create_contract_nli_dataset(split)
    elif task_name == "ecthr":
        dataset = create_ecthr_dataset(split)

    text_column_names = _task_column_names[task_name]
    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name) 

    def tokenize_function(inp):
        # truncates sentences to max_length or pads them to max_length

        first_half = inp[text_column_names[0]]
        #second_half = inp[
        #    text_column_names[1]] if text_column_names[1] in inp else None
        return tokenizer(
            text=first_half,
            #text_pair=second_half,
            padding='max_length',
            max_length=max_seq_length,
            truncation=True,
        )
    
    assert isinstance(dataset, datasets.Dataset)
    dataset = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=None if num_workers == 0 else num_workers,
        batch_size=1000,
        new_fingerprint=f'{task_name}-tok-2-{split}-{max_seq_length}',
        load_from_cache_file=False,
    )

    for column in dataset.features:
        if column not in ['label', 'label_ids', 'input_ids', 'token_type_ids', 'attention_mask']:
            dataset = dataset.remove_columns([column])

    print("Final Columns for Dataset")
    print(dataset.features)

    return dataset