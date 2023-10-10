
import sys
import os
import torch
from tqdm import tqdm
from collections import Counter, defaultdict

from sklearn.metrics import accuracy_score
import transformers

#sys.path.append('/var/cr05_data/sim_data/code/mm-bert-experiments/bert/')
import src.mosaic_bert as mosaic_bert_module
import omegaconf as om



# Pick a Task
TASK = 'mnli'   # Enter your task here

TASK_NAME_TO_NUM_LABELS = {
    'mnli': 3,
    'rte': 2,
    'mrpc': 2,
    'qnli': 2,
    'qqp': 2,
    'sst2': 2,
    'stsb': 1,
    'cola': 2
}

num_labels = TASK_NAME_TO_NUM_LABELS[TASK]




# Load Models 

keys2models = {}

yaml_prefix = "/var/cr05_data/sim_data/code/mm-bert-experiments/bert/yamls/finetuning/glue/"
finetune_prefix = "/var/cr06_data/sim_data/local-finetune-checkpoints/"

paths = {
    "transformer": {
        "expt_path": "mosaic-train-c4-full-mosaic-transformer-mlm0.15-noglu-ep0-ba35000/",
        "yaml_path": "mosaic-bert-base-uncased-noglu.yaml",
    },
    "hybrid": {
        "expt_path": "m2hybrid-c4-noglu-mlm0.15-seqlen128-ep0-ba35000/",
        "yaml_path": "bert-monarch-longconv-hack-add-attn.yaml",
    },
    "m2": {
        "expt_path": "m2-c4-noglu-mlm0.15-seqlen128-AdamW-lr1e-4-ep0-ba42000/",
        "yaml_path": "m2-bert-base-uncased.yaml",
    }
}

for model_key in paths.keys():
    print(f"Loading {model_key} model")
    
    expt_path = f"{finetune_prefix}/{paths[model_key]['expt_path']}"
    yaml_path = f"{yaml_prefix}/{paths[model_key]['yaml_path']}"

    with open(yaml_path) as f:
        yaml_cfg = om.OmegaConf.load(f)
    cli_cfg = om.OmegaConf.from_cli([])
    cfg = om.OmegaConf.merge(yaml_cfg, cli_cfg)

    task = f"/task={TASK}/seed=19/"
    checkpoints = os.listdir(expt_path + task)
    checkpoints = [c for c in checkpoints if c.endswith('.pt') and 'latest' not in c]
    checkpoint_path = expt_path + task + checkpoints[0]

    model = mosaic_bert_module.create_mosaic_bert_classification(
                num_labels=num_labels,
                pretrained_model_name='bert-base-uncased',
                pretrained_checkpoint=None,
                model_config=cfg['model']['model_config'],
                tokenizer_name='bert-base-uncased'
            )
    model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cuda'))['state']['model'])

    keys2models[model_key] = model