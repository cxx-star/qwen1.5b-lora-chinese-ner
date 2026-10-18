from .evaluation import check_entities
import json
import torch


def convert_sample(sample):
    entities = []
    for entity_type,text_to_spans in sample['label'].items():
        for entity_text,spans in text_to_spans.items():
            for start,end in spans:
                entities.append({
                    "type":entity_type,
                    "text":entity_text,
                    "start":start,
                    "end":end+1
                })
    if not check_entities(sample["text"],entities):
        raise ValueError("转换后的实体未通过校验，请检查原文与标注")
    return entities

def load_samples(path):
    samples = []
    with open(path,'r',encoding = 'utf-8') as f:
        line = f.readline()
        n = 0
        while line:
            sample = json.loads(line)
            entities = convert_sample(sample)
            samples.append({"sentence":sample["text"],"gold":entities,"source_index":n})
            n += 1
            line = f.readline()
    return samples


def load_split(samples, split_path):
    with open(split_path,'r',encoding='utf-8') as f:
        d = json.load(f)
        train_indices = d["train_indices"]
        val_indices = d["val_indices"]
    train_samples = []
    val_samples = []
    for i in train_indices:
        train_samples.append(samples[i])
    for i in val_indices:
        val_samples.append(samples[i])
    return train_samples, val_samples

def prepare_training_sample(sample, system_prompt, tokenizer):
    train_messages = [{"role":"system","content":system_prompt},{"role":"user","content":sample["sentence"]},{"role":"assistant","content":json.dumps(sample["gold"], ensure_ascii=False)}]
    prompt_ids = tokenizer.apply_chat_template(train_messages[:2],tokenize=True,add_generation_prompt=True)
    full_ids = tokenizer.apply_chat_template(train_messages,tokenize=True,add_generation_prompt=False)
    assert full_ids[:len(prompt_ids)] == prompt_ids
    labels = full_ids.copy()
    for i in range(len(prompt_ids)):
        labels[i] = -100
    return {"input_ids":full_ids,"labels":labels}

def collate_samples(encoded_samples, pad_token_id):
    max_length = 0
    input_ids = []
    labels = []
    attention_masks = []
    for sample in encoded_samples:
        max_length = max(len(sample["input_ids"]),max_length)
    for sample in encoded_samples:
        input_ids.append(sample["input_ids"] + [pad_token_id] * (max_length - len(sample["input_ids"])))
        labels.append(sample["labels"] + [-100] *  (max_length - len(sample["input_ids"])))
        attention_masks.append([1] * len(sample["input_ids"]) + [0] *  (max_length - len(sample["input_ids"])))
    input_ids = torch.tensor(input_ids,dtype=torch.long,device="cpu")
    labels = torch.tensor(labels,dtype=torch.long,device="cpu")
    attention_masks = torch.tensor(attention_masks,dtype=torch.long,device="cpu")
    return {
        "input_ids":input_ids,
        "labels":labels,
        "attention_mask":attention_masks,
    }
